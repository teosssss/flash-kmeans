# Hopper CUDA Sources

This folder contains the first isolated Hopper flash-assign experiment.

Design reference kernels from `fast.cu`:
- kernel 5: <https://github.com/pranjalssh/fast.cu/blob/main/examples/matmul/matmul_5.cuh>
- kernel 7: <https://github.com/pranjalssh/fast.cu/blob/main/examples/matmul/matmul_7.cuh>

Current experimental kernel:
- `hopper_k5_k7_v1`

The implementation keeps the flash-assign control flow:
- outer loop over centroid tiles
- inner loop over feature tiles
- consumer-side warp minima
- final CTA-level writeback of ids and distances

This version is non-persistent and non-clustered.

Notes on `hopper_k5_k7_wgmma256`:
- `points` and `centroids` stay row-major in global memory. The TMA descriptors plus shared-memory swizzle adapt the tiles for WGMMA; the kernel is not changing the source layout.
- `smem.a_tiles` is the point tile, `smem.b_tiles` is the centroid tile. The comments in the kernel are correct to question the naming, but the intended mapping is A = points, B = centroids.
- `wgmma_b + 0 * kWgmmaK`, `+ 1 * kWgmmaK`, etc. is still the right pattern. WGMMA advances in `K=16` chunks inside the shared tile, not by the full `N=256` width.
- The current `wgmma256` experiment uses `float accum[16][8]` per consumer warpgroup. Each consumer handles one 64-row slice, so the CTA tile is `128x256` instead of keeping two 64-row slices live in one warpgroup.
- The `__syncthreads()` at the end of the kernel is still required before final writeback. It makes the consumer updates to `running_best_*` visible to the CTA before any thread stores output.
