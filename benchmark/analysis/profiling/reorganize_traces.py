#!/usr/bin/env python3
"""Reorganize trace files into phase/mode/run_id hierarchy."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


TRACE_EXTS = {".nsys-rep", ".sqlite", ".qdstrm", ".json"}


def infer_phase_mode_run(path: Path) -> Tuple[str, str, str]:
    name = path.name
    lname = name.lower()
    lfull = str(path).lower()

    phase = "mixed"
    if "prefill" in lname or "prefill" in lfull:
        phase = "prefill"
    elif "decode" in lname or "decode" in lfull:
        phase = "decode"

    mode = "unknown"
    if re.search(r"\btp\d*\b", lname) or lname.startswith("tp_") or "/tp2/" in lfull:
        mode = "tp2"
    elif "dp2_ep" in lname or "dp2ep" in lname or "/dp2_ep/" in lfull:
        mode = "dp2_ep"
    elif re.search(r"\bdp\d*\b", lname) or "/dp2/" in lfull:
        mode = "dp2"

    run_id = re.sub(r"\.[^.]+$", "", name)  # remove one suffix first
    if run_id.endswith(".nsys") or run_id.endswith(".trace"):
        run_id = run_id.rsplit(".", 1)[0]
    run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id).strip("._-")
    if not run_id:
        run_id = "run_unknown"
    if not run_id.startswith("run_"):
        run_id = f"run_{run_id}"
    return phase, mode, run_id


def collect_source_files(traces_root: Path) -> List[Path]:
    files: List[Path] = []
    for p in traces_root.rglob("*"):
        if not p.is_file():
            continue
        if p.name == ".gitkeep":
            continue
        if p.suffix in TRACE_EXTS or p.name.endswith(".trace.json"):
            files.append(p)
    return sorted(files)


def reorganize(traces_root: Path, dry_run: bool = False) -> Dict[str, object]:
    src_files = collect_source_files(traces_root)
    moved = []
    skipped = []

    for src in src_files:
        rel = src.relative_to(traces_root)
        # Skip already standardized paths: <type>/<phase>/<mode>/<run_id>/file
        parts = rel.parts
        if len(parts) >= 5 and parts[0] in {"nsys", "torch"}:
            skipped.append({"path": str(rel), "reason": "already_standardized"})
            continue

        trace_type = "torch" if (src.name.endswith(".trace.json") or "traceevents" in src.name.lower()) else "nsys"
        if src.suffix == ".json" and trace_type != "torch":
            # Heuristic: keep non-trace json out unless it looks like torch trace
            try:
                txt = src.read_text(encoding="utf-8", errors="ignore")
                if '"traceEvents"' in txt:
                    trace_type = "torch"
                else:
                    skipped.append({"path": str(rel), "reason": "json_not_trace"})
                    continue
            except Exception:
                skipped.append({"path": str(rel), "reason": "json_unreadable"})
                continue

        phase, mode, run_id = infer_phase_mode_run(src)
        dst_dir = traces_root / trace_type / phase / mode / run_id
        dst = dst_dir / src.name

        if dst.resolve() == src.resolve():
            skipped.append({"path": str(rel), "reason": "same_target"})
            continue
        if dst.exists():
            # avoid overwrite by appending suffix
            base = dst.stem
            suf = dst.suffix
            i = 2
            while True:
                cand = dst_dir / f"{base}__dup{i}{suf}"
                if not cand.exists():
                    dst = cand
                    break
                i += 1

        moved.append(
            {
                "src": str(rel),
                "dst": str(dst.relative_to(traces_root)),
                "trace_type": trace_type,
                "phase": phase,
                "mode": mode,
                "run_id": run_id,
            }
        )
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    return {"moved": moved, "skipped": skipped}


def maybe_copy_torch_trace_into_traces(repo_root: Path, traces_root: Path, dry_run: bool) -> List[Dict[str, str]]:
    copied: List[Dict[str, str]] = []
    # Known existing torch trace location from this repo
    candidates = sorted((repo_root / "results" / "tables").rglob("trace_len_*.json"))
    for src in candidates:
        try:
            txt = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if '"traceEvents"' not in txt:
            continue
        m = re.search(r"trace_len_(\d+)\.json$", src.name)
        length = m.group(1) if m else "unknown"
        dst_dir = traces_root / "torch" / "prefill" / "unknown" / f"run_len_{length}"
        dst = dst_dir / src.name
        if dst.exists():
            continue
        copied.append({"src": str(src.relative_to(repo_root)), "dst": str(dst.relative_to(repo_root))})
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorganize traces into phase/mode/run_id layout")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--traces-root", default="results/traces")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-json", default="results/analysis/profiling/trace_reorg_report.json")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    traces_root = (repo_root / args.traces_root).resolve()
    traces_root.mkdir(parents=True, exist_ok=True)

    copied = maybe_copy_torch_trace_into_traces(repo_root, traces_root, dry_run=args.dry_run)
    result = reorganize(traces_root, dry_run=args.dry_run)
    result["copied_into_traces"] = copied
    result["dry_run"] = args.dry_run

    out = repo_root / args.report_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[trace reorg] report: {out}")
    print(f"[trace reorg] moved={len(result['moved'])} skipped={len(result['skipped'])} copied={len(copied)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()

