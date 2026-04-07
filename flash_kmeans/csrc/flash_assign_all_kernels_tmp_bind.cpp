#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

cudaError_t launch_point_l2_norm_kernel(
    const half* points,
    float* point_norms,
    int num_points,
    int dim,
    cudaStream_t stream
);

cudaError_t launch_centroid_l2_norm_kernel(
    const half* centroids,
    float* centroid_norms,
    int num_centroids,
    int dim,
    cudaStream_t stream
);

cudaError_t launch_flash_assign_kernel_256x128x32_force_generic(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

cudaError_t launch_flash_assign_kernel_256x128x32_force_aligned(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

cudaError_t launch_flash_assign_kernel_256x128x32_force_aligned_static_k_128(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
);
cudaError_t launch_flash_assign_kernel_256x128x32_force_aligned_static_k_256(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
);
cudaError_t launch_flash_assign_kernel_256x128x32_force_aligned_static_k_512(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

cudaError_t launch_flash_assign_kernel_256x128x32_force_deferred_generic(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

cudaError_t launch_flash_assign_kernel_256x128x32_force_deferred_static_k_128(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
);
cudaError_t launch_flash_assign_kernel_256x128x32_force_deferred_static_k_256(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
);
cudaError_t launch_flash_assign_kernel_256x128x32_force_deferred_static_k_512(
    const half* A,
    const half* B_col_major,
    const float* x_norm,
    const float* c_norm,
    int* output_ids,
    float* output_dists,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

std::vector<torch::Tensor> flash_assign_all_kernels_tmp_cuda(
    torch::Tensor points,
    torch::Tensor centroids,
    std::string kernel_name
) {
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be a CUDA tensor");
    TORCH_CHECK(points.dtype() == torch::kFloat16, "points must be float16");
    TORCH_CHECK(centroids.dtype() == torch::kFloat16, "centroids must be float16");
    TORCH_CHECK(points.dim() == 2, "points must have shape [num_points, dim]");
    TORCH_CHECK(centroids.dim() == 2, "centroids must have shape [num_centroids, dim]");
    TORCH_CHECK(points.size(1) == centroids.size(1), "points and centroids must have the same dim");
    TORCH_CHECK(points.is_contiguous(), "points must be contiguous");
    TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");

    const auto num_points = static_cast<int>(points.size(0));
    const auto num_centroids = static_cast<int>(centroids.size(0));
    const auto dim = static_cast<int>(points.size(1));

    auto point_norms = torch::empty({num_points}, points.options().dtype(torch::kFloat32));
    auto centroid_norms = torch::empty({num_centroids}, centroids.options().dtype(torch::kFloat32));
    auto output_ids = torch::empty({num_points}, points.options().dtype(torch::kInt32));
    auto output_dists = torch::empty({num_points}, points.options().dtype(torch::kFloat32));

    const c10::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(points.device().index()).stream();

    cudaError_t err = launch_point_l2_norm_kernel(
        reinterpret_cast<const half*>(points.data_ptr<at::Half>()),
        point_norms.data_ptr<float>(),
        num_points,
        dim,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "launch_point_l2_norm_kernel failed: ", cudaGetErrorString(err));

    err = launch_centroid_l2_norm_kernel(
        reinterpret_cast<const half*>(centroids.data_ptr<at::Half>()),
        centroid_norms.data_ptr<float>(),
        num_centroids,
        dim,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "launch_centroid_l2_norm_kernel failed: ", cudaGetErrorString(err));

    const half* a_ptr = reinterpret_cast<const half*>(points.data_ptr<at::Half>());
    const half* b_ptr = reinterpret_cast<const half*>(centroids.data_ptr<at::Half>());

    if (kernel_name == "generic_main") {
        err = launch_flash_assign_kernel_256x128x32_force_generic(
            a_ptr, b_ptr, point_norms.data_ptr<float>(), centroid_norms.data_ptr<float>(),
            output_ids.data_ptr<int>(), output_dists.data_ptr<float>(),
            num_points, num_centroids, dim, stream
        );
    } else if (kernel_name == "aligned_generic_main") {
        err = launch_flash_assign_kernel_256x128x32_force_aligned(
            a_ptr, b_ptr, point_norms.data_ptr<float>(), centroid_norms.data_ptr<float>(),
            output_ids.data_ptr<int>(), output_dists.data_ptr<float>(),
            num_points, num_centroids, dim, stream
        );
    } else if (kernel_name == "aligned_static_main") {
        TORCH_CHECK(dim == 128 || dim == 256 || dim == 512, "aligned_static_main requires K in {128,256,512}");
        if (dim == 128) {
            err = launch_flash_assign_kernel_256x128x32_force_aligned_static_k_128(
                a_ptr, b_ptr, point_norms.data_ptr<float>(), centroid_norms.data_ptr<float>(),
                output_ids.data_ptr<int>(), output_dists.data_ptr<float>(),
                num_points, num_centroids, dim, stream
            );
        } else if (dim == 256) {
            err = launch_flash_assign_kernel_256x128x32_force_aligned_static_k_256(
                a_ptr, b_ptr, point_norms.data_ptr<float>(), centroid_norms.data_ptr<float>(),
                output_ids.data_ptr<int>(), output_dists.data_ptr<float>(),
                num_points, num_centroids, dim, stream
            );
        } else {
            err = launch_flash_assign_kernel_256x128x32_force_aligned_static_k_512(
                a_ptr, b_ptr, point_norms.data_ptr<float>(), centroid_norms.data_ptr<float>(),
                output_ids.data_ptr<int>(), output_dists.data_ptr<float>(),
                num_points, num_centroids, dim, stream
            );
        }
    } else if (kernel_name == "deferred_generic") {
        err = launch_flash_assign_kernel_256x128x32_force_deferred_generic(
            a_ptr, b_ptr, point_norms.data_ptr<float>(), centroid_norms.data_ptr<float>(),
            output_ids.data_ptr<int>(), output_dists.data_ptr<float>(),
            num_points, num_centroids, dim, stream
        );
    } else if (kernel_name == "deferred_static") {
        TORCH_CHECK(dim == 128 || dim == 256 || dim == 512, "deferred_static requires K in {128,256,512}");
        if (dim == 128) {
            err = launch_flash_assign_kernel_256x128x32_force_deferred_static_k_128(
                a_ptr, b_ptr, point_norms.data_ptr<float>(), centroid_norms.data_ptr<float>(),
                output_ids.data_ptr<int>(), output_dists.data_ptr<float>(),
                num_points, num_centroids, dim, stream
            );
        } else if (dim == 256) {
            err = launch_flash_assign_kernel_256x128x32_force_deferred_static_k_256(
                a_ptr, b_ptr, point_norms.data_ptr<float>(), centroid_norms.data_ptr<float>(),
                output_ids.data_ptr<int>(), output_dists.data_ptr<float>(),
                num_points, num_centroids, dim, stream
            );
        } else {
            err = launch_flash_assign_kernel_256x128x32_force_deferred_static_k_512(
                a_ptr, b_ptr, point_norms.data_ptr<float>(), centroid_norms.data_ptr<float>(),
                output_ids.data_ptr<int>(), output_dists.data_ptr<float>(),
                num_points, num_centroids, dim, stream
            );
        }
    } else {
        TORCH_CHECK(false, "Unknown kernel_name: ", kernel_name);
    }

    TORCH_CHECK(err == cudaSuccess, "forced flash assign launch failed: ", cudaGetErrorString(err));
    return {output_ids, output_dists, point_norms, centroid_norms};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_assign_all_kernels_tmp_cuda", &flash_assign_all_kernels_tmp_cuda,
          "Flash assign all kernels tmp");
}
