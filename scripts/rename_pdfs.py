"""Rename research PDF files to:

    <FirstAuthorLastName> - <ShortTitle>.pdf

Heuristics (ordered):
1. If name contains ' - ' and a token that looks like a surname -> use it.
2. If name starts with '<Surname>_' -> take first chunk as surname.
3. If name starts with '<Surname>-' -> same as above.
4. If a space-separated token matches surname pattern -> use following tokens as title.
5. If pattern '<Surname>_<Topic>' (e.g. 'Fok_ACO_based_Tool_path_Optimizer') -> split.

Limitations:
- No PDF metadata parsing (filename only).
- Pure ID filenames (e.g. '1-s2.0-...') left unchanged.
- Unicode normalized (percent-decoded + NFC).

Outputs:
- In-place rename under docs/research.
- CSV map: docs/research/rename_map.csv (original,new,status).

Safety:
- Skips files already matching '<Surname> - <Title>'.

Usage:
    python scripts/rename_pdfs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import unicodedata
from urllib.parse import unquote

RESEARCH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "research")
MAP_PATH = os.path.join(RESEARCH_DIR, "rename_map.csv")

SURNAME_PATTERN = re.compile(r"^[A-Z][a-zA-Z'\-]{1,}$")
STOPWORDS = {
    'International',
    'Journal',
    'Applied',
    'Polymer',
    'Science',
    'Polymers',
    'Electronics',
    'Materials',
    'Mechanical',
    'Process',
    'Processing',
    'Modeling',
    'Models',
    'Parameter',
    'Optimization',
    'Cost',
    'Effective',
    'Cost-effective',
    'Bookseries',
    'BookSeries',
    'MACS',
    'Using',
    'AI',
    'Of',
    'And',
    'For',
    'In',
    'On',
    'An',
    'A',
    'The',
    'J',
    'Conf',
    'Ser',
    'Mater',
}


def normalize_text(s: str) -> str:
    s = unquote(s)  # percent decode
    s = unicodedata.normalize("NFC", s)
    return s


def cleanup_title(raw: str) -> str:
    # Remove extraneous tokens like years (4 digits) when surrounded by separators.
    raw = re.sub(r"[_-]+", " ", raw)
    raw = re.sub(r"\b\d{4}\b", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    # Capitalize nicely but preserve internal capitalization tokens like FDM, ACO
    tokens = []
    for t in raw.split():
        if t.isupper() and len(t) <= 6:  # keep acronyms
            tokens.append(t)
        else:
            tokens.append(t.capitalize())
    title = " ".join(tokens)
    # Truncate overly long titles
    return title[:120].strip()


def is_already_pattern(fname_no_ext: str) -> bool:
    return bool(re.match(r"^[A-Z][a-zA-Z'\-]+ - .+", fname_no_ext))


def extract_author_and_title(base: str) -> tuple[str | None, str | None]:
    # Base is filename without extension
    if is_already_pattern(base):
        return None, None  # signal skip

    b_norm = normalize_text(base)

    # Heuristic 1: pattern with ' - ' containing surname
    if ' - ' in b_norm:
        parts = [p.strip() for p in b_norm.split(' - ') if p.strip()]
        # Prefer surname as the last part: "... - <Surname>"
        if parts and SURNAME_PATTERN.match(parts[-1]) and len(parts) > 1:
            return parts[-1], cleanup_title(' - '.join(parts[:-1]))
        # Or surname as the first part: "<Surname> - <Title>"
        if parts and SURNAME_PATTERN.match(parts[0]) and len(parts) > 1:
            return parts[0], cleanup_title(' - '.join(parts[1:]))

    # Heuristic 2: surname underscore prefix (Titlecase surname)
    if '_' in b_norm:
        first_token, rest = b_norm.split('_', 1)
        # Accept titlecase surnames without internal hyphens and not generic stopwords
        if (
            SURNAME_PATTERN.match(first_token)
            and first_token == first_token.capitalize()
            and '-' not in first_token
            and first_token not in STOPWORDS
        ):
            return first_token, cleanup_title(rest)
        # Trailing pattern _<Surname>_<Year>
        m = re.search(r"_(?P<surname>[A-Z][a-zA-Z'\-]{1,})_(?P<year>\d{4})$", b_norm)
        if m:
            surname = m.group('surname')
            title_raw = b_norm[: m.start()].rstrip('_- ')
            if title_raw:
                return surname, cleanup_title(title_raw)

    # Heuristic 3: surname prefix before first hyphen with rest title (strict)
    if '-' in b_norm:
        first_token, rest = b_norm.split('-', 1)
        if (
            SURNAME_PATTERN.match(first_token)
            and first_token == first_token.capitalize()
            and first_token not in STOPWORDS
        ):
            return first_token, cleanup_title(rest)

    # Heuristic 4: disabled (avoids false positives like generic last tokens)

    # Unable to extract
    return None, None


def planned_new_name(author: str, title: str) -> str:
    safe_title = re.sub(r"[\/:]", " - ", title)
    safe_title = re.sub(r"[?*<>|]", "", safe_title)
    return f"{author} - {safe_title}.pdf"


def main(dry_run: bool = False):
    rows = []
    for fname in sorted(os.listdir(RESEARCH_DIR)):
        if not fname.lower().endswith('.pdf'):
            continue
        old_path = os.path.join(RESEARCH_DIR, fname)
        base = fname[:-4]
        author, title = extract_author_and_title(base)
        if author is None and title is None:
            rows.append([fname, fname, 'skip-pattern-or-unknown'])
            continue
        if author is None or title is None:
            rows.append([fname, fname, 'insufficient-metadata'])
            continue
        new_fname = planned_new_name(author, title)
        new_path = os.path.join(RESEARCH_DIR, new_fname)
        if os.path.exists(new_path):
            rows.append([fname, new_fname, 'target-exists'])
            continue
        rows.append([fname, new_fname, 'renamed' if not dry_run else 'dry-run'])
        if not dry_run:
            try:
                os.rename(old_path, new_path)
            except OSError as e:
                rows[-1][-1] = f'error:{e.__class__.__name__}'

    # Write map CSV
    with open(MAP_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['original', 'new', 'status'])
        w.writerows(rows)

    print(f"Processed {len(rows)} PDF files. Map written to {MAP_PATH}")
    counters = {}
    for r in rows:
        counters[r[2]] = counters.get(r[2], 0) + 1
    print("Status counts:")
    for k, v in sorted(counters.items()):
        print(f"  {k}: {v}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Rename research PDFs heuristically extracting first author surname '
            'and short title from filename.'
        )
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='Do not rename; only produce mapping file.'
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
