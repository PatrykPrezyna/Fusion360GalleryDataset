import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Set


def get_categories_from_assembly(assembly_path: str) -> List[str]:
    """Return list of categories from an assembly.json file."""
    try:
        with open(assembly_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    props = data.get("properties", {})
    cats = props.get("categories", [])

    # Normalise to list of strings
    if isinstance(cats, str):
        cats = [cats]
    elif not isinstance(cats, list):
        cats = []

    # Clean up category names a bit
    clean = []
    for c in cats:
        if isinstance(c, str):
            c = c.strip()
            if c:
                clean.append(c)
    return clean


def step_part_paths_in_dir(dir_path: str) -> Set[str]:
    """
    Count STEP part files in a single design directory.

    We treat all *.step files as parts, except any whose basename starts with
    'assembly' (i.e. assembly.step), which we consider assembly files.
    """
    parts: Set[str] = set()
    for name in os.listdir(dir_path):
        if not name.lower().endswith(".step"):
            continue
        base = os.path.splitext(name)[0].lower()
        if base.startswith("assembly"):
            # Skip assembly STEP files
            continue
        parts.add(os.path.abspath(os.path.join(dir_path, name)))
    return parts


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.join(repo_root, "input_data")

    # Track unique part STEP files per category (by absolute path)
    category_parts: Dict[str, Set[str]] = defaultdict(set)
    # Track all unique part STEP files overall
    all_parts: Set[str] = set()

    # Walk through all subdirectories under input_data
    for root, dirs, files in os.walk(input_root):
        if "assembly.json" not in files:
            continue

        assembly_path = os.path.join(root, "assembly.json")
        categories = get_categories_from_assembly(assembly_path)

        # STEP parts in the same directory as this assembly.json
        part_paths = step_part_paths_in_dir(root)
        if not part_paths:
            continue

        # Update global set of all parts (avoid double-counting across dirs)
        all_parts.update(part_paths)

        # Update per-category sets
        if not categories:
            # If no category information, bucket under "Uncategorized"
            category_parts["Uncategorized"].update(part_paths)
        else:
            for cat in categories:
                category_parts[cat].update(part_paths)

    # Prepare output structure
    # Convert sets to counts, sorted by category name
    category_counts = {
        cat: len(paths) for cat, paths in sorted(category_parts.items(), key=lambda x: x[0])
    }

    stats = {
        "total_unique_parts": len(all_parts),
        "categories": category_counts,
    }

    out_path = os.path.join(repo_root, "statistics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(f"Saved statistics to {out_path}")


if __name__ == "__main__":
    main()


