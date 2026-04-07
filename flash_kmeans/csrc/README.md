# CUDA source layout

- `flash_assign.cu`: assign kernels and launchers.
- `flash_assign_common.cuh`: shared tile constants, `cp.async` helpers, WMMA helpers, and shared-memory staging utilities.
- `flash_assign_norm_kernels.cuh`: row-wise L2 norm kernel used by both points and centroids.
- `flash_assign_bind.cpp`: default PyTorch extension entry point.
- `flash_assign_all_kernels_tmp_bind.cpp`: benchmark-only entry point that exposes multiple CUDA kernel variants.

The temporary `*_tmp.cu` files are variant experiments. The main comparison path for CUDA vs Triton should go through `flash_assign.cu` plus `flash_assign_all_kernels_tmp_bind.cpp` so all variants are measured through one consistent extension surface.
