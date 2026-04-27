"""Step 1.2 — Build the bundled SQLite database from the upstream YAML clone.

Reads `refractiveindex.info-database/database/catalog-nk.yml`, parses every
referenced material YAML, and writes a SQLite file at
`src/indicio/data/refractiveindex.db`.

`tabulated n2` blocks (Kerr nonlinear index) are skipped — out of scope for
v0.1. Materials whose only DATA blocks are n2 are dropped entirely.

Run:
    uv run python scripts/build_database.py
"""

from __future__ import annotations

import struct
import subprocess
import sys
import zlib
from collections import Counter
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = REPO_ROOT / "refractiveindex.info-database"
DATA_DIR = UPSTREAM / "database" / "data"
CATALOG_PATH = UPSTREAM / "database" / "catalog-nk.yml"
DB_DIR = REPO_ROOT / "src" / "indicio" / "data"
DB_PATH = DB_DIR / "refractiveindex.db"


SCHEMA = """
CREATE TABLE refs (
    ref_id  INTEGER PRIMARY KEY,
    text    TEXT NOT NULL UNIQUE
);

CREATE TABLE materials (
    shelf       TEXT NOT NULL,
    book        TEXT NOT NULL,
    page        TEXT NOT NULL,
    name        TEXT,
    ref_id      INTEGER,
    comments    TEXT,
    PRIMARY KEY (shelf, book, page),
    FOREIGN KEY (ref_id) REFERENCES refs(ref_id)
);

CREATE TABLE models (
    shelf       TEXT NOT NULL,
    book        TEXT NOT NULL,
    page        TEXT NOT NULL,
    quantity    TEXT NOT NULL,
    model_kind  TEXT NOT NULL,
    payload     BLOB NOT NULL,
    n_points    INTEGER,
    wl_min      REAL,
    wl_max      REAL,
    PRIMARY KEY (shelf, book, page, quantity),
    FOREIGN KEY (shelf, book, page) REFERENCES materials(shelf, book, page)
);

CREATE INDEX idx_book ON materials(book);

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def get_upstream_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def iter_catalog(path: Path):
    """Yield (shelf, book, page, name, data_path) tuples from a catalog yaml."""
    with path.open("r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    for shelf_entry in catalog:
        if not isinstance(shelf_entry, dict) or "SHELF" not in shelf_entry:
            continue
        shelf = shelf_entry["SHELF"]
        for book_entry in shelf_entry.get("content", []) or []:
            if not isinstance(book_entry, dict) or "BOOK" not in book_entry:
                continue
            book = book_entry["BOOK"]
            for page_entry in book_entry.get("content", []) or []:
                if not isinstance(page_entry, dict) or "PAGE" not in page_entry:
                    continue
                page = page_entry["PAGE"]
                name = page_entry.get("name")
                data_rel = page_entry.get("data")
                if data_rel is None:
                    continue
                yield shelf, book, page, name, DATA_DIR / data_rel


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split()]


def parse_tabulated(text: str, ncols: int) -> list[list[float]]:
    cols: list[list[float]] = [[] for _ in range(ncols)]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != ncols:
            raise ValueError(
                f"expected {ncols} columns, got {len(parts)}: {line!r}"
            )
        for i, p in enumerate(parts):
            cols[i].append(float(p))
    return cols


def pack_floats(values) -> bytes:
    """Pack values as little-endian IEEE 754 float32 (4 bytes each)."""
    return struct.pack(f"<{len(values)}f", *values)


def compress_tabulated(payload: bytes) -> bytes:
    return zlib.compress(payload, level=9)


def encode_block(block: dict):
    """Convert one DATA block into a list of model rows.

    Returns a list of (quantity, model_kind, payload, n_points, wl_min, wl_max).
    Returns [] for `tabulated n2` (skipped).

    Encoding:
      * All floats are packed as little-endian float32.
      * Tabulated payloads are zlib-compressed (level 9). Formula payloads
        are stored raw (they are tiny — compression overhead would dwarf gains).
    """
    t = block["type"]

    if t == "tabulated nk":
        wl, n_, k_ = parse_tabulated(block["data"], 3)
        wl_blob = pack_floats(wl)
        n_blob = pack_floats(n_)
        k_blob = pack_floats(k_)
        npts = len(wl)
        wl_min, wl_max = min(wl), max(wl)
        return [
            ("n", "tabulated_n", compress_tabulated(wl_blob + n_blob), npts, wl_min, wl_max),
            ("k", "tabulated_k", compress_tabulated(wl_blob + k_blob), npts, wl_min, wl_max),
        ]

    if t == "tabulated n":
        wl, n_ = parse_tabulated(block["data"], 2)
        npts = len(wl)
        return [
            (
                "n",
                "tabulated_n",
                compress_tabulated(pack_floats(wl) + pack_floats(n_)),
                npts,
                min(wl),
                max(wl),
            )
        ]

    if t == "tabulated k":
        wl, k_ = parse_tabulated(block["data"], 2)
        npts = len(wl)
        return [
            (
                "k",
                "tabulated_k",
                compress_tabulated(pack_floats(wl) + pack_floats(k_)),
                npts,
                min(wl),
                max(wl),
            )
        ]

    if t.startswith("formula "):
        formula_n = int(t.split()[1])
        wl_min = wl_max = None
        wr = block.get("wavelength_range")
        if wr is not None:
            mn, mx = parse_floats(wr)
            wl_min, wl_max = mn, mx
        coeffs = parse_floats(block["coefficients"])
        return [
            (
                "n",
                f"formula_{formula_n}",
                pack_floats(coeffs),
                None,
                wl_min,
                wl_max,
            )
        ]

    if t == "tabulated n2":
        return []

    raise ValueError(f"unknown DATA type: {t!r}")


def build():
    if not CATALOG_PATH.is_file():
        print(f"catalog not found: {CATALOG_PATH}", file=sys.stderr)
        return 1

    commit = get_upstream_commit()
    print(f"upstream commit: {commit}")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    conn.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    conn.executescript(SCHEMA)

    raw_entries = list(iter_catalog(CATALOG_PATH))
    # Upstream catalogue contains a few duplicate (shelf, book, page) entries
    # that point to the same data file (display-time redundancy). Dedupe
    # those silently; abort if the same key points at conflicting files.
    seen: dict[tuple[str, str, str], tuple] = {}
    duplicate_count = 0
    for entry in raw_entries:
        shelf, book, page, name, data_path = entry
        key = (shelf, book, page)
        if key in seen:
            existing = seen[key]
            if existing[4] != data_path:
                raise RuntimeError(
                    f"catalog conflict for {key}: {existing[4]} vs {data_path}"
                )
            duplicate_count += 1
            continue
        seen[key] = entry
    catalog_entries = list(seen.values())
    print(f"catalog entries: {len(raw_entries)} raw, "
          f"{len(catalog_entries)} unique ({duplicate_count} duplicates merged)")

    materials_rows: list[tuple] = []
    model_rows: list[tuple] = []
    ref_intern: dict[str, int] = {}

    def intern_ref(text: str | None) -> int | None:
        if text is None:
            return None
        rid = ref_intern.get(text)
        if rid is None:
            rid = len(ref_intern) + 1
            ref_intern[text] = rid
        return rid

    skipped_n2_only = 0
    skipped_missing_file = 0
    encode_failures: list[tuple[tuple[str, str, str], str]] = []
    quantity_collisions: list[tuple[str, str, str]] = []
    kind_counter: Counter[str] = Counter()

    total = len(catalog_entries)
    is_tty = sys.stderr.isatty()
    for i, (shelf, book, page, name, data_path) in enumerate(catalog_entries, start=1):
        if is_tty:
            sys.stderr.write(f"\r[{i:>4}/{total}] {shelf}/{book}/{page}\033[K")
            sys.stderr.flush()
        elif i % 500 == 0 or i == total:
            print(f"  [{i}/{total}] {shelf}/{book}/{page}", flush=True)

        if not data_path.is_file():
            skipped_missing_file += 1
            continue

        try:
            with data_path.open("r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except Exception as exc:
            encode_failures.append(((shelf, book, page), f"yaml: {exc!r}"))
            continue

        data_blocks = doc.get("DATA") or []

        per_quantity: dict[str, tuple] = {}
        try:
            for block in data_blocks:
                for quantity, kind, payload, npts, wmn, wmx in encode_block(block):
                    if quantity in per_quantity:
                        # Upstream redundancy: a few materials provide both a
                        # formula and a tabulated representation for `n`.
                        # First wins (formulas come first in upstream order
                        # and are exact; tabulated would just be a sampling).
                        quantity_collisions.append((shelf, book, page))
                        continue
                    per_quantity[quantity] = (kind, payload, npts, wmn, wmx)
        except Exception as exc:
            encode_failures.append(((shelf, book, page), repr(exc)))
            continue

        if not per_quantity:
            skipped_n2_only += 1
            continue

        materials_rows.append(
            (
                shelf,
                book,
                page,
                name,
                intern_ref(doc.get("REFERENCES")),
                doc.get("COMMENTS"),
            )
        )
        for quantity, (kind, payload, npts, wmn, wmx) in per_quantity.items():
            model_rows.append(
                (shelf, book, page, quantity, kind, payload, npts, wmn, wmx)
            )
            kind_counter[kind] += 1

    if is_tty:
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    with conn:
        conn.executemany(
            "INSERT INTO refs VALUES (?, ?)",
            [(rid, text) for text, rid in ref_intern.items()],
        )
        conn.executemany(
            "INSERT INTO materials VALUES (?, ?, ?, ?, ?, ?)", materials_rows
        )
        conn.executemany(
            "INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", model_rows
        )
        conn.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [("upstream_commit", commit), ("schema_version", "1")],
        )

    conn.commit()
    size_before_vacuum = DB_PATH.stat().st_size
    conn.execute("VACUUM")
    conn.commit()
    size_after_vacuum = DB_PATH.stat().st_size
    conn.close()

    # Smoke checks (1.4)
    print()
    print("=== build summary ===")
    print(f"materials inserted:       {len(materials_rows)}")
    print(f"model rows inserted:      {len(model_rows)}")
    print(f"unique reference texts:   {len(ref_intern)}")
    print(f"skipped (n2 only):        {skipped_n2_only}")
    print(f"skipped (missing file):   {skipped_missing_file}")
    print(f"encode failures:          {len(encode_failures)}")
    for key, why in encode_failures[:10]:
        print(f"  {key}: {why}")
    if len(encode_failures) > 10:
        print(f"  ... and {len(encode_failures) - 10} more")
    print(f"quantity collisions (first wins): {len(quantity_collisions)}")
    for key in quantity_collisions[:10]:
        print(f"  {key}")

    print()
    print("=== model_kind distribution ===")
    width = max((len(k) for k in kind_counter), default=10)
    for k, n in kind_counter.most_common():
        print(f"  {k:<{width}}  {n:>6}")

    print()
    print("=== file size ===")
    print(f"after insert:   {size_before_vacuum / 1024**2:7.2f} MiB  ({size_before_vacuum:>10} bytes)")
    print(f"after VACUUM:   {size_after_vacuum / 1024**2:7.2f} MiB  ({size_after_vacuum:>10} bytes)")

    # Payload-size breakdown by kind (tabulated rows are already zlib-compressed).
    conn2 = sqlite3.connect(DB_PATH)
    bytes_by_kind: Counter[str] = Counter()
    for kind, payload_len in conn2.execute(
        "SELECT model_kind, length(payload) FROM models"
    ):
        bytes_by_kind[kind] += payload_len
    conn2.close()
    total_payload = sum(bytes_by_kind.values())
    print(f"total payload bytes: {total_payload / 1024**2:7.2f} MiB ({total_payload} bytes)")
    width = max((len(k) for k in bytes_by_kind), default=10)
    for kind, n in bytes_by_kind.most_common():
        print(f"  {kind:<{width}}  {n / 1024**2:7.2f} MiB  ({n} bytes)")

    print()
    print("Run `uv run pytest` to verify integrity and spot-check materials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
