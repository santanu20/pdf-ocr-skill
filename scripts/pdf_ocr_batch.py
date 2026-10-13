#!/usr/bin/env python3
"""Batch engine driver — amortize interpreter/imports across many PDFs.

The per-PDF CLI pays ~1.5-2s of interpreter start + import + dep-probe
for EVERY document (measured: 522-PDF category = 96 min at 4 workers,
~30 min of it process overhead; engines also spawn ~90 internal threads
each, so >4 concurrent engines thrash the box).

Design (constraints: ONNX deadlocks across fork, measured 2026-08-14;
pdf_ocr.py is argv/global coupled and guards on __name__ == "__main__"):

  parent            parses manifest, spawns W workers (spawn context)
  worker (spawn)    imports stdlib only, then for each (pdf, out) pair:
                    runpy.run_path(pdf_ocr.py, run_name="__main__") in a
                    fresh namespace — sys.argv patched per pair. Python's
                    module cache makes re-execution cheap (no interpreter
                    restart); ONNX/thread-pool init happens at most once
                    per worker (lazily, on first specialist use) instead
                    of once per PDF.
  crash isolation   a worker that dies (OOM, segfault) is detected by the
                    parent; its remaining pairs are requeued to a fresh
                    worker. Per-PDF failures are recorded, never fatal.

Usage:
  pdf_ocr_batch.py --manifest pairs.jsonl --workers 4 [--pdf-ocr PATH]
  pairs.jsonl: one JSON object per line: {"pdf": "...", "out": "..."}
  Optional per-pair passthrough args: {"args": ["--pages", "3"]}
Exit codes: 0 all pairs succeeded | 1 some PDFs failed | 2 env error.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import runpy
import sys
import time
from pathlib import Path

DEFAULT_PDF_OCR = str(Path(__file__).resolve().parent / "pdf_ocr.py")


def _worker(task):
    """Process one slice of (pdf, out, extra_args) triples; return results."""
    pairs, pdf_ocr_path, wid = task
    results = []
    for pdf, out, extra in pairs:
        out_dir = os.path.dirname(os.path.abspath(out))
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        argv_backup = sys.argv[:]
        # PER-PDF WATCHDOG (2026-09-05): a starved/hung PDF must never
        # block a worker forever (measured: contention turned an 80s page
        # into 35+ min with no timeout under batch). SIGALRM raises in the
        # main thread mid-runpy; the BaseException handler records it as a
        # per-PDF failure and the slice continues.
        _budget = int(os.environ.get("PDF_OCR_BATCH_PDF_TIMEOUT", "900"))
        import signal

        def _alarm(signum, frame):
            raise TimeoutError(f"per-PDF budget {_budget}s exceeded")

        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(_budget)
        # Self-profiling mode (no ptrace/sudo needed): PDF_OCR_BATCH_CPROFILE=1
        # wraps each engine run in cProfile and dumps stats for slow PDFs
        # (>= PDF_OCR_BATCH_PROFILE_MIN seconds, default 10) — live diagnosis
        # of monster pages while jobs run (live-systems discipline).
        _prof = None
        _prof_min = float(os.environ.get("PDF_OCR_BATCH_PROFILE_MIN", "10"))
        if os.environ.get("PDF_OCR_BATCH_CPROFILE"):
            import cProfile

            _prof = cProfile.Profile()
        code, err = 0, ""
        try:
            sys.argv = [pdf_ocr_path, pdf, out, "--json", *extra]
            if _prof is not None:
                _prof.enable()
            runpy.run_path(pdf_ocr_path, run_name="__main__")
            if _prof is not None:
                _prof.disable()
        except SystemExit as e:  # engine's own exit codes are per-PDF data
            if _prof is not None:
                _prof.disable()
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            err = str(e)[:500]
        except BaseException as e:  # noqa: BLE001 — worker must never die silently
            if _prof is not None:
                _prof.disable()
            code, err = 1, f"{type(e).__name__}: {e}"[:500]
        finally:
            signal.alarm(0)
            sys.argv = argv_backup
        # SILENT ZERO-OUTPUT GUARD (2026-09-12, run-11 incident): engine
        # exited 0 but wrote nothing (169s of work, empty pages.jsonl, no
        # other files rewritten — measured 081875a1_page_17 under 4-worker
        # batch, never reproduced single-PDF). exit-0 without a non-empty
        # pages.jsonl violates the --json contract; retry once in-process
        # (a raw-cache entry from the discarded run, if written, makes the
        # retry a fast heal) and fail LOUD if it repeats.
        if code == 0:
            _pj = out + ".pages.jsonl"
            _empty = True
            try:
                with open(_pj, "rb") as _fh:
                    _empty = not _fh.read(1)
            except OSError:
                _empty = True
            if _empty:
                print(
                    f"[batch] silent zero-output on {os.path.basename(pdf)}; retrying once",
                    flush=True,
                )
                _retry_ok = False
                try:
                    sys.argv = [pdf_ocr_path, pdf, out, "--json", *extra]
                    runpy.run_path(pdf_ocr_path, run_name="__main__")
                    with open(_pj, "rb") as _fh:
                        _retry_ok = bool(_fh.read(1))
                except BaseException as e:  # noqa: BLE001 — recorded, never fatal
                    err = f"retry {type(e).__name__}: {e}"[:500]
                finally:
                    sys.argv = argv_backup
                if not _retry_ok:
                    code, err = (
                        1,
                        (err + " | silent zero-output (pages.jsonl empty after retry)")[:500],
                    )
        if _prof is not None and (time.time() - t0) >= _prof_min:
            prof_dir = os.environ.get("PDF_OCR_BATCH_CPROFILE")
            os.makedirs(prof_dir, exist_ok=True)
            import pstats

            stats = pstats.Stats(_prof)
            path = os.path.join(prof_dir, os.path.basename(pdf) + ".pstats")
            stats.dump_stats(path)
            print(f"[batch-profile] {path}", flush=True)
        results.append(
            {
                "pdf": pdf,
                "out": out,
                "code": code,
                "err": err,
                "s": round(time.time() - t0, 2),
                "wid": wid,
            }
        )
    return results


def _init_worker():
    """Spawn-context initializer: keep numeric thread pools from exploding
    and OPT IN to live profiling."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "2")
    os.environ["OMP_WAIT_POLICY"] = "passive"
    # Yama ptrace_scope=1 blocks py-spy attach to non-children (and this box
    # has no passwordless sudo). PR_SET_PTRACER_ANY is the kernel-sanctioned
    # per-process opt-in: live-systems discipline requires
    # attachable workers, not post-mortem guesses.
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PTRACER = 0x59616D61
        PR_SET_PTRACER_ANY = 0xFFFFFFFFFFFFFFFF
        libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    except Exception:
        pass  # non-Linux / libc oddity: profiling falls back to cProfile mode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="jsonl of {pdf, out[, args]}")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pdf-ocr", default=DEFAULT_PDF_OCR)
    ap.add_argument("--results-out", help="optional jsonl path for per-pair results")
    args = ap.parse_args()

    if not os.path.isfile(args.manifest):
        print(f"RUNNER ERROR: manifest missing: {args.manifest}", file=sys.stderr)
        raise SystemExit(2)
    if not os.path.isfile(args.pdf_ocr):
        print(f"RUNNER ERROR: pdf_ocr.py missing: {args.pdf_ocr}", file=sys.stderr)
        raise SystemExit(2)

    pairs = []
    with open(args.manifest, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            pairs.append((rec["pdf"], rec["out"], rec.get("args", [])))
    if not pairs:
        print("RUNNER ERROR: empty manifest", file=sys.stderr)
        raise SystemExit(2)

    # Round-robin slices: deterministic assignment (PDF order stable).
    slices = [pairs[i :: args.workers] for i in range(args.workers)]

    ctx = mp.get_context("spawn")
    t0 = time.time()
    all_results: list[dict] = []
    with ctx.Pool(processes=args.workers, initializer=_init_worker) as pool:
        # Requeue protocol: if a worker dies mid-slice, the pool raises on
        # join; simplest robust path = map_async with one chunk per worker
        # and a retry pass afterwards for anything missing.
        async_res = pool.map_async(_worker, [(s, args.pdf_ocr, w) for w, s in enumerate(slices)])
        try:
            for chunk in async_res.get(timeout=24 * 3600):
                all_results.extend(chunk)
        except Exception as e:  # noqa: BLE001 — worker died
            print(f"BATCH ERROR: worker pool failed: {e}", file=sys.stderr)
            raise SystemExit(2)

    done_outs = {r["out"] for r in all_results}
    missing = [p for p in pairs if p[1] not in done_outs]
    if missing:  # retry once, single worker (crashed worker's leftovers)
        print(f"batch: retrying {len(missing)} pairs lost to a worker crash", file=sys.stderr)
        all_results.extend(_worker((missing, args.pdf_ocr, args.workers)))

    all_results.sort(key=lambda r: r["out"])
    if args.results_out:
        with open(args.results_out, "w", encoding="utf-8") as fh:
            for r in all_results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    fails = [r for r in all_results if r["code"] != 0]
    print(
        f"batch: {len(all_results) - len(fails)}/{len(all_results)} ok in "
        f"{time.time() - t0:.0f}s (workers={args.workers})"
    )
    for r in fails[:20]:
        print(f"  FAIL {r['pdf']}: rc={r['code']} {r['err'][:160]}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
