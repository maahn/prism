#!/usr/bin/env python3
"""Move date-named files that sit directly in a product folder into YYYY/
subfolders, so every product folder of a data archive uses the layout PRISM
writes (<site>/cloudnet/<product>/<YYYY>/<YYYYMMDD>_*.nc).

    yearly_folders.py ROOT [ROOT ...]          # dry run: only prints what would move
    yearly_folders.py --apply ROOT [ROOT ...]  # actually move

ROOT is a site's cloudnet directory. Only files named YYYYMMDD_* directly
inside a first-level folder are touched; README/LICENSE and existing year
folders are left alone. Moves are same-filesystem renames; an existing
destination is never overwritten.
"""
import argparse
import re
import sys
from pathlib import Path

DATED = re.compile(r"^(\d{4})\d{4}_")


def plan(root: Path):
    for product in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(product.iterdir()):
            m = DATED.match(f.name)
            if f.is_file() and m:
                yield f, product / m.group(1) / f.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--apply", action="store_true", help="move the files (default: dry run)")
    args = ap.parse_args()
    moved = skipped = 0
    per_folder: dict[Path, int] = {}
    for root in args.roots:
        for src, dst in plan(root):
            if dst.exists():
                print(f"SKIP (destination exists): {src}", file=sys.stderr)
                skipped += 1
                continue
            per_folder[src.parent] = per_folder.get(src.parent, 0) + 1
            if args.apply:
                dst.parent.mkdir(exist_ok=True)
                src.rename(dst)
            moved += 1
    for folder, n in per_folder.items():
        print(f"{folder}: {n} file(s)")
    print(f"{'moved' if args.apply else 'would move'} {moved} file(s), skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
