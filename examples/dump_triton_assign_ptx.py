import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/flash_kmeans_triton_cache")

import torch

from flash_kmeans.assign_euclid_triton import euclid_assign_triton, _heuristic_euclid_config


def parse_shape(shape):
    parts = tuple(int(x) for x in shape.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected M,N,K")
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, default=(16384, 1024, 128))
    parser.add_argument("--max-lines", type=int, default=120)
    args = parser.parse_args()

    cache = Path(os.environ["TRITON_CACHE_DIR"])
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True, exist_ok=True)

    m, n, k = args.shape
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    points = torch.randn((m, k), generator=gen, device="cuda", dtype=torch.float16).contiguous()
    centroids = torch.randn((n, k), generator=gen, device="cuda", dtype=torch.float16).contiguous()

    x = points.unsqueeze(0)
    c = centroids.unsqueeze(0)
    x_sq = (x.float() * x.float()).sum(dim=-1)
    c_sq = (c.float() * c.float()).sum(dim=-1)

    config = _heuristic_euclid_config(m, n, k, device=points.device)
    print(f"shape=({m},{n},{k}) config={config}")
    out = euclid_assign_triton(x, c, x_sq, c_sq=c_sq)
    torch.cuda.synchronize()
    print(f"out_checksum={int(out.sum().item())}")

    files = sorted(p for p in cache.rglob("*") if p.is_file())
    print(f"cache_files={len(files)}")
    for path in files:
        rel = path.relative_to(cache)
        print(f"CACHE_FILE {rel} bytes={path.stat().st_size}")

    interesting_suffixes = {".ttir", ".ttgir", ".llir", ".ptx"}
    for path in files:
        if path.suffix not in interesting_suffixes:
            continue
        text = path.read_text(errors="ignore")
        rel = path.relative_to(cache)
        lowered = text.lower()
        print(f"\n===== {rel} =====")
        print(
            "COUNTS "
            f"mma={lowered.count('mma')} "
            f"wgmma={lowered.count('wgmma')} "
            f"ldmatrix={lowered.count('ldmatrix')} "
            f"cp_async={lowered.count('cp.async')} "
            f"dot={lowered.count('dot')} "
            f"shared={lowered.count('.shared')} "
            f"global_load={lowered.count('ld.global')} "
            f"global_store={lowered.count('st.global')}"
        )
        lines = text.splitlines()
        selected = []
        needles = (
            "tt.dot",
            "dot",
            "mma",
            "wgmma",
            "ldmatrix",
            "cp.async",
            "ld.global",
            "st.global",
            "for ",
            "scf.for",
            "bra",
        )
        for idx, line in enumerate(lines, start=1):
            low = line.lower()
            if any(needle in low for needle in needles):
                selected.append(f"{idx}: {line}")
        for line in selected[: args.max_lines]:
            print(line)
        if len(selected) > args.max_lines:
            print(f"... truncated {len(selected) - args.max_lines} matching lines")


if __name__ == "__main__":
    main()
