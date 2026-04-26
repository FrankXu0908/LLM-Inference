#!/usr/bin/env python3
"""Classify trace artifacts and emit an inventory report."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _is_torch_trace_json(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, dict) and isinstance(data.get("traceEvents"), list)
    except Exception:
        return False


def _sqlite_status(path: Path) -> str:
    try:
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        finally:
            conn.close()
        return "valid"
    except Exception:
        return "invalid"


def _collect_trace_files(root: Path) -> List[Path]:
    exts = {".nsys-rep", ".sqlite", ".qdstrm", ".json", ".trace.json"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix in exts or p.name.endswith(".trace.json"):
            out.append(p)
    return sorted(out)


def classify(paths: List[Path], repo_root: Path) -> Dict[str, Any]:
    items = []
    counts = defaultdict(int)

    for p in paths:
        rel = p.relative_to(repo_root)
        category = "other"
        status = "ok"
        if p.suffix == ".nsys-rep":
            category = "nsys_raw_rep"
        elif p.suffix == ".qdstrm":
            category = "nsys_raw_qdstrm"
        elif p.suffix == ".sqlite":
            category = "nsys_sqlite_export"
            status = _sqlite_status(p)
        elif p.name.endswith(".trace.json") or _is_torch_trace_json(p):
            category = "torch_trace_json"
        elif p.suffix == ".json":
            category = "json_other"

        counts[f"{category}:{status}"] += 1
        items.append(
            {
                "path": str(rel),
                "category": category,
                "status": status,
                "size_bytes": p.stat().st_size,
            }
        )

    return {"summary": dict(counts), "items": items}


def write_markdown(inv: Dict[str, Any], md_path: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Trace Inventory\n\n")
        f.write("## Summary\n\n")
        for k, v in sorted(inv["summary"].items()):
            f.write(f"- `{k}`: {v}\n")

        f.write("\n## Invalid / Needs Attention\n\n")
        bad = [x for x in inv["items"] if x["status"] != "ok" and x["status"] != "valid"]
        if not bad:
            f.write("- none\n")
        else:
            for it in bad:
                f.write(f"- `{it['path']}` (`{it['category']}`, `{it['status']}`)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify trace files and produce inventory")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--trace-root", default="results/traces")
    parser.add_argument("--include-extra", nargs="*", default=["results/tables"], help="Additional roots to scan for torch trace json")
    parser.add_argument("--output-json", default="results/analysis/profiling/trace_inventory.json")
    parser.add_argument("--output-md", default="results/analysis/profiling/trace_inventory.md")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    scan_roots = [Path(args.trace_root)]
    scan_roots.extend(Path(x) for x in args.include_extra)

    all_paths: List[Path] = []
    for root in scan_roots:
        root_abs = (repo_root / root).resolve() if not root.is_absolute() else root
        if root_abs.exists():
            all_paths.extend(_collect_trace_files(root_abs))

    inv = classify(sorted(set(all_paths)), repo_root)

    out_json = repo_root / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")

    write_markdown(inv, repo_root / args.output_md)
    print(f"[trace inventory] saved json: {out_json}")
    print(f"[trace inventory] saved md:   {repo_root / args.output_md}")


if __name__ == "__main__":
    main()

