#!/usr/bin/env python3
"""
Ingest filament property data from extracted CSV tables into materials.json.
Handles range parsing, unit normalization, and schema extension.
"""

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

# Paths
TABLES_DIR = Path(__file__).parent.parent / "data" / "raw" / "filament_tables"
MATERIALS_FILE = Path(__file__).parent.parent / "materials.json"
BACKUP_FILE = MATERIALS_FILE.with_suffix(".json.backup")


def parse_range(value: str) -> dict[str, float | None]:
    """Parse a range string like '55 - 75' or single value."""
    if pd.isna(value) or value == "N/A" or not value:
        return {"min": None, "max": None, "typical": None}

    value_str = str(value).strip()

    # Handle ranges with newlines (from CSV extraction)
    value_str = re.sub(r'\s*\n\s*', ' ', value_str)

    # Try range pattern: "55 - 75" or "2100 - 3600"
    range_match = re.search(r'([\d.]+)\s*-\s*([\d.]+)', value_str)
    if range_match:
        min_val = float(range_match.group(1))
        max_val = float(range_match.group(2))
        return {"min": min_val, "max": max_val, "typical": (min_val + max_val) / 2}

    # Try single numeric value
    single_match = re.search(r'([\d.]+)', value_str)
    if single_match:
        val = float(single_match.group(1))
        return {"min": val, "max": val, "typical": val}

    return {"min": None, "max": None, "typical": None}


def clean_material_name(name: str) -> str:
    """Normalize material names to match existing keys."""
    name = str(name).strip().upper()

    # Handle common variants
    mapping = {
        "PEI (ULTEM 9085)": "ULTEM_9085",
        "TPE (E.G., 85A)": "TPE_85A",
        "TPU (E.G., 95A)": "TPU_95A",
        "BASIC PLA": "PLA",
        "GLASS FIBER PLA": "PLA_GF",
        "CARBON FIBER PLA": "PLA_CF",
        "CF-PETG (10 WT%)": "PETG_CF",
        "NYLON (PA)": "NYLON",
        "CF-NYLON": "NYLON_CF",
    }

    for pattern, result in mapping.items():
        if pattern in name:
            return result

    # Default: remove special chars and spaces
    return re.sub(r'[^A-Z0-9]', '_', name).strip('_')


def ingest_standard_materials():
    """Ingest standard materials from table_2_page_5.csv."""
    csv_path = TABLES_DIR / "table_2_page_5.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path.name} not found")
        return {}

    df = pd.read_csv(csv_path)
    materials = {}

    for _, row in df.iterrows():
        mat_name = clean_material_name(row["Material"])

        strength = parse_range(row.get("Tensile\nStrength\n(MPa)", ""))
        modulus = parse_range(row.get("Tensile\nModulus\n(MPa)", ""))
        elongation = parse_range(row.get("Elongati\non at\nBreak\n(%)", ""))
        hdt = parse_range(row.get("HDT /\nMax\nService\nTemp\n(°C)", ""))
        impact = parse_range(row.get("Impact\nStrength\n(kJ/m²)", ""))
        density = parse_range(row.get("Density\n(g/cm³)", ""))

        materials[mat_name] = {
            "tensile_strength_MPa": strength["typical"],
            "tensile_modulus_MPa": modulus["typical"],
            "elongation_at_break_pct": elongation["typical"],
            "HDT_C": hdt["typical"],
            "impact_strength_kJ_m2": impact["typical"],
            "density_g_cm3": density["typical"],
        }

    return materials


def ingest_high_performance_materials():
    """Ingest high-performance materials from table_5_page_6.csv."""
    csv_path = TABLES_DIR / "table_5_page_6.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path.name} not found")
        return {}

    df = pd.read_csv(csv_path)
    materials = {}

    for _, row in df.iterrows():
        mat_name = clean_material_name(row["Material"])

        strength = parse_range(row.get("Tensile\nStrength\n(MPa)", ""))
        modulus = parse_range(row.get("Tensile\nModulus\n(MPa)", ""))
        elongation = parse_range(row.get("Elongati\non at\nBreak\n(%)", ""))
        hdt = parse_range(row.get("HDT /\nMax\nService\nTemp\n(°C)", ""))
        tg = parse_range(row.get("Glass\nTransitio\nn (Tg)\n(°C)", ""))
        density = parse_range(row.get("Density\n(g/cm³)", ""))

        materials[mat_name] = {
            "tensile_strength_MPa": strength["typical"],
            "tensile_modulus_MPa": modulus["typical"],
            "elongation_at_break_pct": elongation["typical"],
            "HDT_C": hdt["typical"],
            "Tg_C": tg["typical"],
            "density_g_cm3": density["typical"],
        }

    return materials


def ingest_flexible_materials():
    """Ingest flexible materials from table_6_page_6.csv."""
    csv_path = TABLES_DIR / "table_6_page_6.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path.name} not found")
        return {}

    df = pd.read_csv(csv_path)
    materials = {}

    for _, row in df.iterrows():
        mat_name = clean_material_name(row["Material"])

        # Parse Shore hardness (e.g., "85A")
        shore_str = str(row.get("Shore\nHardness", ""))
        shore_match = re.search(r'(\d+)([A-D])', shore_str)
        shore_hardness = shore_match.group(0) if shore_match else None

        strength = parse_range(row.get("Tensile\nStrength\n(MPa)", ""))
        elongation = parse_range(row.get("Elongation\nat Break\n(%)", ""))
        density = parse_range(row.get("Density\n(g/cm³)", ""))

        materials[mat_name] = {
            "shore_hardness": shore_hardness,
            "tensile_strength_MPa": strength["typical"],
            "elongation_at_break_pct": elongation["typical"],
            "density_g_cm3": density["typical"],
        }

    return materials


def merge_materials(
    existing: dict[str, Any], new_data: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Merge new material data into existing materials.json structure."""
    for mat_key, new_props in new_data.items():
        if mat_key not in existing:
            # New material: create with defaults
            existing[mat_key] = {
                "common": {
                    "nozzle_temperature": None,
                    "bed_temperature": None,
                    "print_speed": None,
                    "fan_speed": None,
                },
                "properties": {},
            }

        # Ensure properties section exists
        if "properties" not in existing[mat_key]:
            existing[mat_key]["properties"] = {}

        # Merge new properties (don't overwrite existing)
        for prop_key, prop_val in new_props.items():
            if prop_val is not None:
                existing[mat_key]["properties"][prop_key] = prop_val

    return existing


def generate_diff_report(old_materials: dict[str, Any], new_materials: dict[str, Any]) -> str:
    """Generate a human-readable diff report."""
    lines = ["Materials Ingestion Report", "=" * 40, ""]

    old_keys = set(old_materials.keys())
    new_keys = set(new_materials.keys())

    added = new_keys - old_keys
    if added:
        lines.append(f"✓ Added {len(added)} new materials:")
        for key in sorted(added):
            lines.append(f"  + {key}")
        lines.append("")

    updated_count = 0
    for key in sorted(old_keys & new_keys):
        old_props = old_materials[key].get("properties", {})
        new_props = new_materials[key].get("properties", {})

        if old_props != new_props:
            updated_count += 1

    if updated_count:
        lines.append(f"✓ Updated {updated_count} existing materials")
        lines.append("")

    lines.append(f"Total materials: {len(new_materials)}")

    return "\n".join(lines)


def main():
    """Main ingestion workflow."""
    print("Starting material data ingestion...")

    # Load existing materials
    if not MATERIALS_FILE.exists():
        print(f"Error: {MATERIALS_FILE} not found")
        return

    with open(MATERIALS_FILE) as f:
        existing_materials = json.load(f)

    # Backup
    with open(BACKUP_FILE, 'w') as f:
        json.dump(existing_materials, f, indent=2)
    print(f"✓ Backed up to {BACKUP_FILE.name}")

    # Ingest from all tables
    print("\nIngesting data from CSV tables...")

    standard = ingest_standard_materials()
    print(f"  Standard materials: {len(standard)}")

    high_perf = ingest_high_performance_materials()
    print(f"  High-performance materials: {len(high_perf)}")

    flexible = ingest_flexible_materials()
    print(f"  Flexible materials: {len(flexible)}")

    # Combine all ingested data
    all_new = {}
    for data_dict in [standard, high_perf, flexible]:
        all_new.update(data_dict)

    print(f"\nTotal new/updated entries: {len(all_new)}")

    # Merge with existing
    merged = merge_materials(existing_materials.copy(), all_new)

    # Generate diff
    diff_report = generate_diff_report(existing_materials, merged)
    print("\n" + diff_report)

    # Save updated materials
    with open(MATERIALS_FILE, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\n✓ Updated {MATERIALS_FILE.name}")
    print(f"  Backup available at: {BACKUP_FILE.name}")


if __name__ == "__main__":
    main()
