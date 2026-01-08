#!/usr/bin/env python3

from __future__ import annotations
import os, subprocess, shutil, sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class GPUInfo:
    index: int
    free_gb: float
    total_gb: float
    util_pct: Optional[float] = None

def _query_with_pynvml() -> List[GPUInfo]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return []
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        out: List[GPUInfo] = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                pass
            out.append(GPUInfo(
                index=i,
                free_gb=mem.free / (1024**3),
                total_gb=mem.total / (1024**3),
                util_pct=float(util) if util is not None else None
            ))
        return out
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

def _query_with_nvidia_smi() -> List[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    fmt = "--query-gpu=index,memory.free,memory.total,utilization.gpu"
    cmd = ["nvidia-smi", fmt, "--format=csv,noheader,nounits"]
    try:
        txt = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return []
    out: List[GPUInfo] = []
    for line in txt.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:  # index, free, total[, util]
            continue
        try:
            idx = int(parts[0])
            free_gb = float(parts[1]) / 1024.0 if float(parts[1]) > 256 else float(parts[1])  # handle MiB or GiB
            total_gb = float(parts[2]) / 1024.0 if float(parts[2]) > 256 else float(parts[2])
            util = float(parts[3]) if len(parts) > 3 and parts[3] not in ("N/A", "") else None
            out.append(GPUInfo(index=idx, free_gb=free_gb, total_gb=total_gb, util_pct=util))
        except Exception:
            continue
    return out

def list_gpus() -> List[GPUInfo]:
    gpus = _query_with_pynvml()
    if gpus:
        return gpus
    return _query_with_nvidia_smi()

def pick_freest_gpu(
    gpus: List[GPUInfo],
    min_free_gb: float = 1.0,
    exclude: Optional[List[int]] = None,
    max_util_pct: Optional[float] = None,
) -> Optional[GPUInfo]:
    exclude = set(exclude or [])
    candidates = []
    for g in gpus:
        if g.index in exclude:
            continue
        if g.free_gb < min_free_gb:
            continue
        if (max_util_pct is not None) and (g.util_pct is not None) and (g.util_pct > max_util_pct):
            continue
        candidates.append(g)
    if not candidates:
        # No GPU met the threshold; pick emptiest overall as a fallback
        return max(gpus, key=lambda x: x.free_gb) if gpus else None
    return max(candidates, key=lambda x: x.free_gb)

def mask_gpu(index: int) -> None:
    # IMPORTANT: Call BEFORE importing torch or creating any CUDA objects
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)

def select_and_mask_gpu(
    min_free_gb: float = 1.0,
    exclude: Optional[List[int]] = None,
    max_util_pct: Optional[float] = None,
    set_visible: bool = True,
    strict: bool = False,
) -> Optional[int]:
    gpus = list_gpus()
    if not gpus:
        if strict:
            return None
        else:
            return None  # No GPUs visible; caller can fall back to CPU

    chosen = pick_freest_gpu(gpus, min_free_gb=min_free_gb, exclude=exclude, max_util_pct=max_util_pct)
    if chosen is None:
        if strict:
            return None
        return None

    if set_visible:
        mask_gpu(chosen.index)
    return chosen.index

def _fmt_gpu(g: GPUInfo) -> str:
    util = f"{g.util_pct:.0f}%" if g.util_pct is not None else "N/A"
    return f"cuda:{g.index}  free={g.free_gb:.1f}GB / {g.total_gb:.1f}GB  util={util}"

def main(argv: List[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Select the freest NVIDIA GPU and optionally set CUDA_VISIBLE_DEVICES.")
    p.add_argument("--min-free-gb", type=float, default=1.0, help="Minimum free memory to consider a GPU (GB).")
    p.add_argument("--exclude", type=int, nargs="*", default=None, help="GPU indices to exclude.")
    p.add_argument("--max-util-pct", type=float, default=None, help="Skip GPUs with utilization above this percent.")
    p.add_argument("--set-env", action="store_true", help="Set CUDA_VISIBLE_DEVICES to the chosen GPU.")
    p.add_argument("--print-only", action="store_true", help="Only print selection; do not set env.")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if no GPU meets filters.")
    args = p.parse_args(argv)

    gpus = list_gpus()
    if not gpus:
        print("No GPUs detected (pynvml/nvidia-smi not available or no NVIDIA GPUs).")
        return 1 if args.strict else 0

    print("GPU inventory:")
    for g in gpus:
        print("  " + _fmt_gpu(g))

    chosen = pick_freest_gpu(gpus, min_free_gb=args.min_free_gb, exclude=args.exclude, max_util_pct=args.max_util_pct)
    if chosen is None:
        print("No suitable GPU found.")
        return 1 if args.strict else 0

    print(f"\nSelected: {_fmt_gpu(chosen)}")
    if args.set_env and not args.print_only:
        mask_gpu(chosen.index)
        print(f"CUDA_VISIBLE_DEVICES set to {chosen.index}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))