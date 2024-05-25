#include <iostream>
#include <cassert>
#include <vector>
#include <utility>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cuda_pipeline.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/types.h>
#include <torch/extension.h>

using namespace torch::indexing;
using namespace nvcuda;

#define FULL_MASK 0xffffffff
#define HALF_MASK 0x0000ffff

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) 	        do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(false)
#define gpuErrchk(ans)          do { gpuAssert((ans), __FILE__, __LINE__); } while (false)


__host__ static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}

__device__ static inline uint32_t add_as_half2(uint32_t x, uint32_t y) {
    uint32_t z;
    asm("add.f16x2 %0,%1,%2;" : "=r"(z) : "r"(x), "r"(y));
    return z;
}


__device__ static inline uint32_t mask_lop3(uint32_t x, uint32_t m0, uint32_t m1) {
    uint32_t y;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(y) : "r"(x), "r"(m0), "r"(m1));
    return y;
    // return (x & m0) | m1;
}

#define BASE_OFFSET 0xd080d080
#define XMASK 0x00f000f0
#define WMASK 0x50085008


__global__ static void
// __launch_bounds__(1024, 1024)
decode_matvec_e8p_kernel(
    float *__restrict__ output,
    const uint2 *__restrict__ input,
    const uint2 *__restrict__ weights_compressed,
    const uint32_t *__restrict__ codebook_abs,
    int N,
    int K
) {
    int warpId = threadIdx.y;
    int laneId = threadIdx.x;

    // __shared__ float sum_scratch[16*32];

    // __shared__ uint32_t codebook_local[256*32];
    // for (int icb = warpId; icb < 256; icb += 32) {
    //     codebook_local[icb*32 + laneId] = codebook_abs[icb];
    // }
    // __syncthreads();

    __shared__ uint2 shared_weights[1024*2];

    for (int iin = blockIdx.x; iin < (N >> 4); iin += gridDim.x) {

        float z0 = 0.0;
        float z1 = 0.0;
        float z2 = 0.0;
        float z3 = 0.0;

        // int shwo = laneId + 32*warpId;

        // __pipeline_memcpy_async(shared_weights + shwo, weights_compressed + laneId + 32*warpId + 1024*0 + (K >> 1)*iin, 8);
        // __pipeline_commit();

        for (int iik = warpId; iik < (K >> 6); iik += 32) {
            // if (iik + 1 < (K >> 11)) {
            //     __pipeline_memcpy_async(shared_weights + (shwo ^ 1024), weights_compressed + laneId + 32*iik + 1024 + (K >> 1)*iin, 8);
            //     __pipeline_commit();
            //     __pipeline_wait_prior(1);
            //     shwo = shwo ^ 1024;
            // }
            // else {
            //     __pipeline_wait_prior(0);
            // }

            // uint2 w_compr = shared_weights[shwo]; // weights_compressed[laneId + 32*warpId + 1024*iik + (K >> 1)*iin];
            uint2 w_compr = weights_compressed[laneId + 32*iik + (K >> 1)*iin];
            uint32_t a = w_compr.x;
            uint32_t b = w_compr.y;

            uint32_t s = b;
            s = s ^ (s >> 4);
            s = s ^ (s >> 8);
            s = s ^ (s >> 16);
            uint32_t sb = (s & 15);
            s = b ^ sb;
            sb = sb | (sb << 16);

            uint32_t input_to_warp = ((const uint32_t*)(&input[16*iik]))[laneId];
            uint32_t shifted_laneId = (laneId & 3) << 3;

            /// BLOCK 01
            {
            uint32_t x = codebook_abs[(a >> 0) & 255];
            x = x ^ ((s & 0x11111111) * 14);

            uint32_t o = BASE_OFFSET | ((sb & 0x00010001) << 4);

            uint32_t w00 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
            uint32_t w01 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
            uint32_t w02 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
            uint32_t w03 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

            x = codebook_abs[(a >> 8) & 255];
            x = x ^ ((s & 0x22222222) * 7);

            o = BASE_OFFSET | ((sb & 0x00020002) << 3);
            
            uint32_t w10 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
            uint32_t w11 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
            uint32_t w12 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
            uint32_t w13 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

            // uint2 x_in = input[0 + (laneId & 3)*4 + 16*warpId + 16*32*iik];
            // uint32_t x_in0 = x_in.x;
            // uint32_t x_in1 = x_in.y;

            uint32_t x_in0 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 0);
            uint32_t x_in1 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 1);

            asm(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 },"
                " { %4, %5, %6, %7 },"
                " { %8, %9 },"
                " { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w00), "r"(w10), "r"(w01),  "r"(w11),
                  "r"(x_in0), "r"(x_in1)
            );


            // x_in = input[1 + (laneId & 3)*4 + 16*warpId + 16*32*iik];
            // x_in0 = x_in.x;
            // x_in1 = x_in.y;

            x_in0 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 2);
            x_in1 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 3);

            asm(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 },"
                " { %4, %5, %6, %7 },"
                " { %8, %9 },"
                " { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w02), "r"(w12), "r"(w03), "r"(w13),
                  "r"(x_in0), "r"(x_in1)
            );
            }
            /// BLOCK 23 
            {
            uint32_t x = codebook_abs[(a >> 16) & 255];
            s = s >> 2;
            x = x ^ ((s & 0x11111111) * 14);

            uint32_t o = BASE_OFFSET | ((sb & 0x00040004) << 2);
            
            uint32_t w00 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
            uint32_t w01 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
            uint32_t w02 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
            uint32_t w03 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

            x = codebook_abs[(a >> 24) & 255];
            x = x ^ ((s & 0x22222222) * 7);

            o = BASE_OFFSET | ((sb & 0x00080008) << 1); 

            uint32_t w10 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
            uint32_t w11 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
            uint32_t w12 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
            uint32_t w13 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);


            // uint2 x_in = input[2 + (laneId & 3)*4 + 16*warpId + 16*32*iik];
            // uint32_t x_in0 = x_in.x;
            // uint32_t x_in1 = x_in.y;

            uint32_t x_in0 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 4);
            uint32_t x_in1 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 5);

            asm(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 },"
                " { %4, %5, %6, %7 },"
                " { %8, %9 },"
                " { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w00), "r"(w10), "r"(w01), "r"(w11),
                  "r"(x_in0), "r"(x_in1)
            );


            // x_in = input[3 + (laneId & 3)*4 + 16*warpId + 16*32*iik];
            // x_in0 = x_in.x;
            // x_in1 = x_in.y;

            x_in0 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 6);
            x_in1 = __shfl_sync(FULL_MASK, input_to_warp, shifted_laneId | 7);

            asm(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                " { %0, %1, %2, %3 },"
                " { %4, %5, %6, %7 },"
                " { %8, %9 },"
                " { %0, %1, %2, %3 };"
                : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
                : "r"(w02), "r"(w12), "r"(w03), "r"(w13),
                  "r"(x_in0), "r"(x_in1)
            );
            }
        }

        // we produced 16 outputs, so only 16 threads
        if ((laneId & 1) == 0) {
            atomicAdd(output + (iin << 4) + (laneId >> 1), (laneId & 2) ? z2 : z0);
        }

        // if ((laneId & 3) == 0) {
        //     sum_scratch[warpId + ((laneId >> 1) + 0) * 32] = z0;
        //     sum_scratch[warpId + ((laneId >> 1) + 1) * 32] = z2;
        // }
        // __syncthreads();

        // // load and sum
        // if (warpId < 16) {
        //     float acc = sum_scratch[laneId + warpId*32];
        //     for (int offset = 16; offset > 0; offset /= 2) {
        //         acc += __shfl_down_sync(FULL_MASK, acc, offset);
        //     }
        //     if (laneId == 0) {
        //         output[(iin << 4) + warpId] = acc;
        //     }
        // }
    }
}


__host__ extern torch::Tensor decode_matvec_e8p(
    torch::Tensor x,
    torch::Tensor weights_compressed,
    torch::Tensor codebook_abs
) {

    CHECK_INPUT(x);
    CHECK_INPUT(weights_compressed);
    CHECK_INPUT(codebook_abs);

    TORCH_CHECK(x.dim() == 1);
    TORCH_CHECK(weights_compressed.dim() == 4);
    TORCH_CHECK(weights_compressed.size(3) == 4);
    TORCH_CHECK(weights_compressed.size(2) == 8);
    TORCH_CHECK(codebook_abs.dim() == 1);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);
    TORCH_CHECK(weights_compressed.scalar_type() == torch::kInt64);
    TORCH_CHECK(codebook_abs.scalar_type() == torch::kInt32);
    TORCH_CHECK(x.size(-1) == weights_compressed.size(1) << 6);
    TORCH_CHECK(codebook_abs.size(-1) == 256);

    int64_t N = weights_compressed.size(0) * 16;
    int64_t K = x.size(-1);

    TORCH_CHECK(K % 64 == 0, "K is not divisible by 64");
    TORCH_CHECK(N % 16 == 0, "N is not divisible by 16");

    TORCH_CHECK(K < 65536, "K is not too large");
    TORCH_CHECK(N < 65536, "N is not too large");

    at::DeviceGuard guard(x.device());
    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    torch::Tensor output = torch::zeros(std::vector<int64_t>{N}, options);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, x.get_device());
    int64_t grid_size = static_cast<int64_t>(deviceProp.multiProcessorCount);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    const dim3 block_size(32,32);

    decode_matvec_e8p_kernel<<<grid_size, block_size, 0, stream>>>(
        output.data_ptr<float>(),
        (const uint2*)x.data_ptr<c10::Half>(),
        (const uint2*)weights_compressed.data_ptr<int64_t>(),
        (const uint32_t*)codebook_abs.data_ptr<int32_t>(),
        N,
        K);
    
    gpuErrchk(cudaPeekAtLastError());

    return output;
}



__global__ static void
test_tc_kernel(float *__restrict__ output) {
    int laneId = threadIdx.x;

    uint32_t w0 = (laneId == 0) ? 0x3C003C00 : 0x00000000;
    uint32_t w1 = 0x00000000;
    uint32_t w2 = 0x00000000;
    uint32_t w3 = 0x00000000;

    uint32_t x0 = (laneId == 0) ? 0x3C003C00 : 0x00000000;
    uint32_t x1 = 0x00000000;

    float z0 = 0.0;
    float z1 = 0.0;
    float z2 = 0.0;
    float z3 = 0.0;

    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " { %0, %1, %2, %3 },"
        " { %4, %5, %6, %7 },"
        " { %8, %9 },"
        " { %0, %1, %2, %3 };"
        : "+f"(z0), "+f"(z1), "+f"(z2), "+f"(z3)
        : "r"(w0), "r"(w1), "r"(w2), "r"(w3),
          "r"(x0), "r"(x1)
    );

    output[laneId*4 + 0] = z0;
    output[laneId*4 + 1] = z1;
    output[laneId*4 + 2] = z2;
    output[laneId*4 + 3] = z3;
}

__host__ extern torch::Tensor test_tc() {

    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    torch::Tensor output = torch::zeros(std::vector<int64_t>{32*4}, options);

    test_tc_kernel<<<1, 32>>>(output.data_ptr<float>());
    
    gpuErrchk(cudaPeekAtLastError());

    return output;
}




__global__ static void
test_codebook_expand_kernel(uint32_t *__restrict__ output, const uint32_t *__restrict__ codebook_abs) {
    uint32_t a = threadIdx.x;
    uint32_t b = 0;

    for (int i = 0; i < 8; i++) {
        b |= (((blockIdx.x >> i) & 1) << (4*i));
    }

    uint32_t s = b;
    s = s ^ (s >> 4);
    s = s ^ (s >> 8);
    s = s ^ (s >> 16);
    uint32_t sb = (s & 15);
    s = b ^ sb;
    sb = sb | (sb << 16);

    uint32_t x = codebook_abs[(a >> 0) & 255];
    x = x ^ ((s & 0x11111111) * 14);

    uint32_t o = BASE_OFFSET | ((sb & 0x00010001) << 4);

    uint32_t w0 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
    uint32_t w1 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
    uint32_t w2 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
    uint32_t w3 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

    output[blockIdx.x*256*4 + threadIdx.x*4 + 0] = w0;
    output[blockIdx.x*256*4 + threadIdx.x*4 + 1] = w1;
    output[blockIdx.x*256*4 + threadIdx.x*4 + 2] = w2;
    output[blockIdx.x*256*4 + threadIdx.x*4 + 3] = w3;
}

__host__ extern torch::Tensor test_codebook_expand(torch::Tensor codebook_abs) {

    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    torch::Tensor output = torch::zeros(std::vector<int64_t>{256*256,8}, options);

    test_codebook_expand_kernel<<<256, 256>>>((uint32_t*)output.data_ptr<c10::Half>(), (const uint32_t*)codebook_abs.data_ptr<int32_t>());
    
    gpuErrchk(cudaPeekAtLastError());

    return output;
}




__global__ static void
// __launch_bounds__(1024, 1024)
decompress_packed_e8p_kernel(
    uint32_t *__restrict__ output,
    const uint2 *__restrict__ weights_compressed,
    const uint32_t *__restrict__ codebook_abs,
    int N,
    int K
) {
    int warpId = threadIdx.y;
    int laneId = threadIdx.x;

    for (int iin = blockIdx.x; iin < (N >> 4); iin += gridDim.x) {

        for (int iik = warpId; iik < (K >> 6); iik += 32) {
            uint2 w_compr = weights_compressed[laneId + 32*iik + (K >> 1)*iin];
            uint32_t a = w_compr.x;
            uint32_t b = w_compr.y;

            uint32_t s = b;
            s = s ^ (s >> 4);
            s = s ^ (s >> 8);
            s = s ^ (s >> 16);
            uint32_t sb = (s & 15);
            s = b ^ sb;
            sb = sb | (sb << 16);

            /// BLOCK 01
            {
            uint32_t x = codebook_abs[(a >> 0) & 255];
            x = x ^ ((s & 0x11111111) * 14);

            uint32_t o = BASE_OFFSET | ((sb & 0x00010001) << 4);

            uint32_t w00 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
            uint32_t w01 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
            uint32_t w02 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
            uint32_t w03 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

            x = codebook_abs[(a >> 8) & 255];
            x = x ^ ((s & 0x22222222) * 7);

            o = BASE_OFFSET | ((sb & 0x00020002) << 3);
            
            uint32_t w10 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
            uint32_t w11 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
            uint32_t w12 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
            uint32_t w13 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

            output[iin*8*K + (laneId >> 2)*K + 0 * (K >> 1) + iik*32 + 0*4 + ((laneId & 3) << 3) + 0] = w00;
            output[iin*8*K + (laneId >> 2)*K + 0 * (K >> 1) + iik*32 + 0*4 + ((laneId & 3) << 3) + 1] = w01;
            output[iin*8*K + (laneId >> 2)*K + 1 * (K >> 1) + iik*32 + 0*4 + ((laneId & 3) << 3) + 0] = w10;
            output[iin*8*K + (laneId >> 2)*K + 1 * (K >> 1) + iik*32 + 0*4 + ((laneId & 3) << 3) + 1] = w11;

            output[iin*8*K + (laneId >> 2)*K + 0 * (K >> 1) + iik*32 + 0*4 + ((laneId & 3) << 3) + 2] = w02;
            output[iin*8*K + (laneId >> 2)*K + 0 * (K >> 1) + iik*32 + 0*4 + ((laneId & 3) << 3) + 3] = w03;
            output[iin*8*K + (laneId >> 2)*K + 1 * (K >> 1) + iik*32 + 0*4 + ((laneId & 3) << 3) + 2] = w12;
            output[iin*8*K + (laneId >> 2)*K + 1 * (K >> 1) + iik*32 + 0*4 + ((laneId & 3) << 3) + 3] = w13;

            }
            /// BLOCK 23 
            {
            uint32_t x = codebook_abs[(a >> 16) & 255];
            s = s >> 2;
            x = x ^ ((s & 0x11111111) * 14);

            uint32_t o = BASE_OFFSET | ((sb & 0x00040004) << 2);
            
            uint32_t w00 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
            uint32_t w01 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
            uint32_t w02 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
            uint32_t w03 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

            x = codebook_abs[(a >> 24) & 255];
            x = x ^ ((s & 0x22222222) * 7);

            o = BASE_OFFSET | ((sb & 0x00080008) << 1); 

            uint32_t w10 = add_as_half2(mask_lop3(x << 4, XMASK, WMASK), o);
            uint32_t w11 = add_as_half2(mask_lop3(x << 0, XMASK, WMASK), o);
            uint32_t w12 = add_as_half2(mask_lop3(x >> 4, XMASK, WMASK), o);
            uint32_t w13 = add_as_half2(mask_lop3(x >> 8, XMASK, WMASK), o);

            output[iin*8*K + (laneId >> 2)*K + 0 * (K >> 1) + iik*32 + 1*4 + ((laneId & 3) << 3) + 0] = w00;
            output[iin*8*K + (laneId >> 2)*K + 0 * (K >> 1) + iik*32 + 1*4 + ((laneId & 3) << 3) + 1] = w01;
            output[iin*8*K + (laneId >> 2)*K + 1 * (K >> 1) + iik*32 + 1*4 + ((laneId & 3) << 3) + 0] = w10;
            output[iin*8*K + (laneId >> 2)*K + 1 * (K >> 1) + iik*32 + 1*4 + ((laneId & 3) << 3) + 1] = w11;

            output[iin*8*K + (laneId >> 2)*K + 0 * (K >> 1) + iik*32 + 1*4 + ((laneId & 3) << 3) + 2] = w02;
            output[iin*8*K + (laneId >> 2)*K + 0 * (K >> 1) + iik*32 + 1*4 + ((laneId & 3) << 3) + 3] = w03;
            output[iin*8*K + (laneId >> 2)*K + 1 * (K >> 1) + iik*32 + 1*4 + ((laneId & 3) << 3) + 2] = w12;
            output[iin*8*K + (laneId >> 2)*K + 1 * (K >> 1) + iik*32 + 1*4 + ((laneId & 3) << 3) + 3] = w13;
            }
        }
    }
}


__host__ extern torch::Tensor decompress_packed_e8p(
    torch::Tensor weights_compressed,
    torch::Tensor codebook_abs
) {
    CHECK_INPUT(weights_compressed);
    CHECK_INPUT(codebook_abs);

    TORCH_CHECK(weights_compressed.dim() == 4);
    TORCH_CHECK(weights_compressed.size(3) == 4);
    TORCH_CHECK(weights_compressed.size(2) == 8);
    TORCH_CHECK(codebook_abs.dim() == 1);
    TORCH_CHECK(weights_compressed.scalar_type() == torch::kInt64);
    TORCH_CHECK(codebook_abs.scalar_type() == torch::kInt32);
    TORCH_CHECK(codebook_abs.size(-1) == 256);

    int64_t N = weights_compressed.size(0) * 16;
    int64_t K = weights_compressed.size(1) << 6;

    TORCH_CHECK(K % 64 == 0, "K is not divisible by 64");
    TORCH_CHECK(N % 16 == 0, "N is not divisible by 16");

    TORCH_CHECK(K < 65536, "K is not too large");
    TORCH_CHECK(N < 65536, "N is not too large");

    at::DeviceGuard guard(codebook_abs.device());
    torch::TensorOptions options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .layout(torch::kStrided)
        .device(torch::kCUDA)
        .requires_grad(false);
    torch::Tensor output = torch::zeros(std::vector<int64_t>{N,K}, options);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, weights_compressed.get_device());
    int64_t grid_size = static_cast<int64_t>(deviceProp.multiProcessorCount);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    const dim3 block_size(32,32);

    decompress_packed_e8p_kernel<<<grid_size, block_size, 0, stream>>>(
        (uint32_t*)output.data_ptr<c10::Half>(),
        (const uint2*)weights_compressed.data_ptr<int64_t>(),
        (const uint32_t*)codebook_abs.data_ptr<int32_t>(),
        N,
        K);
    
    gpuErrchk(cudaPeekAtLastError());

    return output;
}