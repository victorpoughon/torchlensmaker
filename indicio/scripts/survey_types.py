"""Step 1.1 — Survey the upstream `DATA` block types.

Walks every `*.yml` under `database/data/` in the upstream clone, parses the
`DATA:` list, and reports the distribution of `type:` values. Also records a
couple of example file paths per type for spot inspection.

Run:
    uv run python scripts/survey_types.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = REPO_ROOT / "refractiveindex.info-database"
DATA_DIR = UPSTREAM / "database" / "data"


def main() -> int:
    if not DATA_DIR.is_dir():
        print(f"upstream data dir not found: {DATA_DIR}", file=sys.stderr)
        return 1

    yaml_files = sorted(DATA_DIR.rglob("*.yml"))
    print(
        f"scanning {len(yaml_files)} yaml files under {DATA_DIR.relative_to(REPO_ROOT)}"
    )

    type_counts: Counter[str] = Counter()
    type_examples: dict[str, list[str]] = defaultdict(list)
    files_with_no_data = 0
    parse_failures: list[tuple[Path, str]] = []
    blocks_per_file: Counter[int] = Counter()
    quantities_per_file: Counter[tuple[str, ...]] = Counter()

    for path in yaml_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except Exception as exc:
            parse_failures.append((path, repr(exc)))
            continue

        if not isinstance(doc, dict):
            parse_failures.append(
                (path, f"top-level is {type(doc).__name__}, expected dict")
            )
            continue

        data = doc.get("DATA")
        if not data:
            files_with_no_data += 1
            continue

        if not isinstance(data, list):
            parse_failures.append(
                (path, f"DATA is {type(data).__name__}, expected list")
            )
            continue

        blocks_per_file[len(data)] += 1

        seen_types: list[str] = []
        for block in data:
            if not isinstance(block, dict):
                parse_failures.append((path, f"DATA block is {type(block).__name__}"))
                continue
            t = block.get("type")
            if not isinstance(t, str):
                parse_failures.append((path, f"DATA.type is {type(t).__name__}: {t!r}"))
                continue
            type_counts[t] += 1
            seen_types.append(t)
            if len(type_examples[t]) < 3:
                rel = path.relative_to(DATA_DIR)
                type_examples[t].append(str(rel))
        quantities_per_file[tuple(seen_types)] += 1

    print()
    print("=== DATA block type distribution ===")
    width = max((len(t) for t in type_counts), default=10)
    for t, n in type_counts.most_common():
        print(f"  {t:<{width}}  {n:>6}")

    print()
    print("=== examples per type (up to 3) ===")
    for t in sorted(type_counts):
        print(f"  {t}")
        for ex in type_examples[t]:
            print(f"      {ex}")

    print()
    print("=== blocks per file ===")
    for k in sorted(blocks_per_file):
        print(f"  {k} block(s):  {blocks_per_file[k]:>6} files")

    print()
    print("=== top 10 (quantity tuple) shapes ===")
    for shape, n in quantities_per_file.most_common(10):
        print(f"  {n:>6}  {shape}")

    print()
    print(f"files with no DATA block: {files_with_no_data}")
    print(f"parse failures: {len(parse_failures)}")
    for p, why in parse_failures[:10]:
        print(f"  {p.relative_to(DATA_DIR)}: {why}")
    if len(parse_failures) > 10:
        print(f"  ... and {len(parse_failures) - 10} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
