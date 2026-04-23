#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdlib>
#include <cfloat>
#include <cstdint>

namespace flash_kmeans::hopper {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerWarpgroup = 4;
constexpr int kWarpgroupThreads = kWarpSize * kWarpsPerWarpgroup;
constexpr int kProducerWarpgroups = 1;
constexpr int kConsumerWarpgroups = 2;
constexpr int kThreadCount = (kProducerWarpgroups + kConsumerWarpgroups) * kWarpgroupThreads;
constexpr int kTileM = 256;
constexpr int kTileN = 128;
constexpr int kTileK = 64;
constexpr int kStageCount = 3;
constexpr int kWgmmaM = 64;
constexpr int kWgmmaN = 128;
constexpr int kWgmmaK = 16;
constexpr int kConsumerRows = kTileM / kConsumerWarpgroups;
constexpr int kPaddedTileK = 64;

__device__ __forceinline__ uint64_t matrix_descriptor_encode(uint64_t x) {
    return ((x & 0x3ffffULL) >> 4);
}

__device__ __forceinline__ uint64_t make_smem_desc(const half* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(16)) << 16;
    desc |= matrix_descriptor_encode(static_cast<uint64_t>(1024)) << 32;
    desc |= 1ULL << 62;
    return desc;
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma128(float d[8][8], const half* sA, const half* sB) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const uint64_t desc_a = make_smem_desc(sA);
    const uint64_t desc_b = make_smem_desc(sB);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66, %67, %68, %69, %70;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
          "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
          "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
          "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
          "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
          "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
          "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#else
    (void)d;
    (void)sA;
    (void)sB;
#endif
}



template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma256(float d[16][8], const half* sA, const half* sB) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130,    %131,  %132,  %133,  %134;\n"
        "}\n"
        :   "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7]),
            "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[4][4]), "+f"(d[4][5]), "+f"(d[4][6]), "+f"(d[4][7]),
            "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[5][4]), "+f"(d[5][5]), "+f"(d[5][6]), "+f"(d[5][7]),
            "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[6][4]), "+f"(d[6][5]), "+f"(d[6][6]), "+f"(d[6][7]),
            "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[7][4]), "+f"(d[7][5]), "+f"(d[7][6]), "+f"(d[7][7]),
            "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]), "+f"(d[8][3]), "+f"(d[8][4]), "+f"(d[8][5]), "+f"(d[8][6]), "+f"(d[8][7]),
            "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[9][4]), "+f"(d[9][5]), "+f"(d[9][6]), "+f"(d[9][7]),
            "+f"(d[10][0]), "+f"(d[10][1]), "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[10][4]), "+f"(d[10][5]), "+f"(d[10][6]), "+f"(d[10][7]),
            "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]), "+f"(d[11][4]), "+f"(d[11][5]), "+f"(d[11][6]), "+f"(d[11][7]),
            "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[12][4]), "+f"(d[12][5]), "+f"(d[12][6]), "+f"(d[12][7]),
            "+f"(d[13][0]), "+f"(d[13][1]), "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[13][4]), "+f"(d[13][5]), "+f"(d[13][6]), "+f"(d[13][7]),
            "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]), "+f"(d[14][4]), "+f"(d[14][5]), "+f"(d[14][6]), "+f"(d[14][7]),
            "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3]), "+f"(d[15][4]), "+f"(d[15][5]), "+f"(d[15][6]), "+f"(d[15][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
#else
    (void)d;
    (void)sA;
    (void)sB;
#endif
}


template <int N>
__device__ __forceinline__ void warpgroup_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" : : "n"(N) : "memory");
#endif
}

__device__ __forceinline__ void warpgroup_arrive() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.fence.sync.aligned;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void warpgroup_commit_batch() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.commit_group.sync.aligned;\n" : : : "memory");
#endif
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroup_reg_alloc() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#endif
}

template <uint32_t RegCount>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#endif
}

__device__ __forceinline__ void init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" : : "r"(bar_ptr), "r"(thread_count + transaction_count));
#else
    (void)bar;
    (void)thread_count;
    (void)transaction_count;
#endif
}

__device__ __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n" : : "r"(bar_ptr), "r"(bytes));
#else
    (void)bar;
    (void)bytes;
#endif
}

__device__ __forceinline__ void wait(uint64_t* bar, int phase_bit) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :
        : "r"(bar_ptr), "r"(phase_bit));
#else
    (void)bar;
    (void)phase_bit;
#endif
}

__device__ __forceinline__ void arrive(uint64_t* bar, uint32_t count = 1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" : : "r"(bar_ptr), "r"(count) : "memory");
#else
    (void)bar;
    (void)count;
#endif
}

template <typename T>
__device__ __forceinline__ void zero_fill(T* dst, int count) {
    #pragma unroll
    for (int i = 0; i < count; ++i) {
        dst[i] = T{};
    }
}

template <int kWarp = kWarpSize>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

template <int kGroupWidth = 4>
__device__ __forceinline__ void reduce_group_min(float& value, int& index) {
    #pragma unroll
    for (int offset = kGroupWidth / 2; offset > 0; offset >>= 1) {
        const float other_value = __shfl_xor_sync(0xffffffffu, value, offset);
        const int other_index = __shfl_xor_sync(0xffffffffu, index, offset);
        if ((other_value < value) || (other_value == value && other_index < index)) {
            value = other_value;
            index = other_index;
        }
    }
}

template <int BlockMajorSize, int BlockMinorSize>
__host__ inline CUtensorMap create_tensor_map(const half* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map;
    void* gmem_address = const_cast<void*>(reinterpret_cast<const void*>(gmem_ptr));
    uint64_t gmem_prob_shape[5] = {64, static_cast<uint64_t>(global_height), static_cast<uint64_t>(global_width / 64), 1, 1};
    uint64_t gmem_prob_stride[5] = {sizeof(half) * global_width, 64 * sizeof(half), 0, 0, 0};
    uint32_t smem_box_shape[5] = {64, static_cast<uint32_t>(BlockMajorSize), static_cast<uint32_t>(BlockMinorSize / 64), 1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        5,
        gmem_address,
        gmem_prob_shape,
        gmem_prob_stride,
        smem_box_shape,
        smem_box_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        std::abort();
    }
    return tma_map;
}

__device__ __forceinline__ void load_async(half* dst, const void* src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    const uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    const uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    const uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5, 0, 0}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr), "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64)
        : "memory");
#else
    (void)dst;
    (void)src_tma_map;
    (void)bar;
    (void)global_col_idx;
    (void)global_row_idx;
#endif
}

__device__ __forceinline__ bool better_candidate(float lhs_value, int lhs_idx, float rhs_value, int rhs_idx) {
    return (lhs_value < rhs_value);
}

}  // namespace flash_kmeans::hopper
