#!/usr/bin/env python3
"""
Extract tables from "FDM Filament Data Analysis and Tables.pdf" to CSV files.
"""

from pathlib import Path

import pandas as pd
import pdfplumber

# Paths
PDF_PATH = (
    Path(__file__).parent.parent / "docs" / "research" / "FDM Filament Data Analysis and Tables.pdf"
)
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "filament_tables"


def extract_tables_from_pdf(pdf_path: Path, output_dir: Path):
    """Extract all tables from the PDF and save as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        print(f"PDF has {len(pdf.pages)} pages")

        table_count = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            if tables:
                print(f"\nPage {page_num}: Found {len(tables)} table(s)")

                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:  # Skip empty or single-row tables
                        continue

                    table_count += 1

                    # Convert to DataFrame
                    df = pd.DataFrame(table[1:], columns=table[0])

                    # Clean up column names (remove None, empty strings)
                    df.columns = [
                        str(col).strip() if col else f"Col_{i}" for i, col in enumerate(df.columns)
                    ]

                    # Remove completely empty rows
                    df = df.dropna(how='all')

                    # Print preview
                    print(f"  Table {table_idx + 1}: {df.shape[0]} rows x {df.shape[1]} cols")
                    print(
                        f"  Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}"
                    )

                    # Save to CSV
                    output_file = output_dir / f"table_{table_count}_page_{page_num}.csv"
                    df.to_csv(output_file, index=False)
                    print(f"  Saved: {output_file.name}")

        print(f"\n✓ Extracted {table_count} tables total")
        return table_count


if __name__ == "__main__":
    if not PDF_PATH.exists():
        print(f"Error: PDF not found at {PDF_PATH}")
        exit(1)

    print(f"Extracting tables from: {PDF_PATH.name}")
    extract_tables_from_pdf(PDF_PATH, OUTPUT_DIR)
    print(f"\nOutput directory: {OUTPUT_DIR}")
