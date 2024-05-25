#include <iostream>
#include <cassert>
#include <vector>
#include <utility>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

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



__global__ void cuda_lookupmatmul_d4_k8_kernel(
    const c10::Half* __restrict__ X,      // k x n
    const uint8_t* __restrict__ YIs,      // m x (n/4)
    const c10::Half* __restrict__ CB,     // 256 x 4
    c10::Half* __restrict__ Z,            // k x m
    size_t K,
    size_t M,
    size_t N) {

  long m1 = blockIdx.x;
  long k1 = blockIdx.y;

  __shared__ c10::Half Y_cache[32*16];

  wmma::fragment<wmma::matrix_a, 8, 32, 16, __half, wmma::row_major> a;  // 8 x 16
  wmma::fragment<wmma::matrix_b, 8, 32, 16, __half, wmma::col_major> b;  // 32 x 16
  wmma::fragment<wmma::accumulator, 8, 32, 16, __half> c;                // 8 x 32
  fill_fragment(c, __float2half(0.0));

  for (long jn = 0; jn < N / 16; jn++) {
# pragma unroll 4
    for (long r = 0; r < 4; r++) {
      uint8_t yidxs = *(uint8_t*)(YIs + jn*(4*M) + m1*4*32 + threadIdx.x*4 + r);
      ((uint64_t*)Y_cache)[threadIdx.x*4 + r] = ((uint64_t*)CB)[(yidxs & 255)];
    }
    load_matrix_sync(a, (const __half*)(X + 8*N*k1 + 16*jn), N);
    load_matrix_sync(b, (const __half*)Y_cache, 16);
    mma_sync(c, a, b, c);
  }
  
  store_matrix_sync((__half*)(&Z[8*M*k1 + 32*m1]), c, M, wmma::mem_row_major);
}


void lookupmatmul_d4_k8(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
) {
  auto k = X.sizes()[0];
  auto m = YIs.sizes()[0];
  auto n = X.sizes()[1];

  assert(X.dtype() == torch::kFloat16);
  assert(YIs.dtype() == torch::kUInt8);
  assert(CB.dtype() == torch::kFloat16);
  assert(Z.dtype() == torch::kFloat16);

  assert(Z.sizes()[0] == k);
  assert(YIs.sizes()[1] * 4 == n);
  assert(Z.sizes()[1] == m);

  assert(k % 8 == 0); // if you want larger k, use k = 16
  assert(m % 32 == 0);
  assert(n % 16 == 0);

  const dim3 threads(32);
  const dim3 blocks(m/32,k/8);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_lookupmatmul_d4_k8_kernel<<<blocks, threads, 0, stream>>>(
    X.data_ptr<c10::Half>(),
    YIs.data_ptr<uint8_t>(),
    CB.data_ptr<c10::Half>(),
    Z.data_ptr<c10::Half>(),
    k,m,n
  );
}



__global__ void cuda_lookupmatmul_d4_k16_kernel(
    const c10::Half* __restrict__ X,      // k x n
    const uint8_t* __restrict__ YIs,      // m x (n/4)
    const c10::Half* __restrict__ CB,     // 256 x 4
    c10::Half* __restrict__ Z,            // k x m
    size_t K,
    size_t M,
    size_t N) {

  long m1 = blockIdx.x;
  long k1 = blockIdx.y;

  __shared__ c10::Half Y_cache[32*16];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;  
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;   
  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c0;               
  fill_fragment(c0, __float2half(0.0));

  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c1;    
  fill_fragment(c1, __float2half(0.0));

  for (long jn = 0; jn < N / 16; jn++) {
    for (long r = 0; r < 4; r++) {
      uint8_t yidxs = *(uint8_t*)(YIs + jn*(4*M) + m1*4*32 + threadIdx.x*4 + r);
      ((uint64_t*)Y_cache)[threadIdx.x*4 + r] = ((uint64_t*)CB)[(yidxs & 255)];
    }

    load_matrix_sync(a, (const __half*)(X + 16*N*k1 + 16*jn), N);

    load_matrix_sync(b, (const __half*)Y_cache, 16);
    mma_sync(c0, a, b, c0);
    
    load_matrix_sync(b, (const __half*)Y_cache + 16*16, 16);
    mma_sync(c1, a, b, c1);
  }
  
  store_matrix_sync((__half*)(&Z[16*M*k1 + 32*m1 +  0]), c0, M, wmma::mem_row_major);
  store_matrix_sync((__half*)(&Z[16*M*k1 + 32*m1 + 16]), c1, M, wmma::mem_row_major);
}


void lookupmatmul_d4_k16(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
) {
  auto k = X.sizes()[0];
  auto m = YIs.sizes()[0];
  auto n = X.sizes()[1];

  assert(X.dtype() == torch::kFloat16);
  assert(YIs.dtype() == torch::kUInt8);
  assert(CB.dtype() == torch::kFloat16);
  assert(Z.dtype() == torch::kFloat16);

  assert(Z.sizes()[0] == k);
  assert(YIs.sizes()[1] * 4 == n);
  assert(Z.sizes()[1] == m);

  assert(k % 16 == 0);
  assert(m % 32 == 0);
  assert(n % 16 == 0);

  const dim3 threads(32);
  const dim3 blocks(m/32,k/16);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_lookupmatmul_d4_k16_kernel<<<blocks, threads, 0, stream>>>(
    X.data_ptr<c10::Half>(),
    YIs.data_ptr<uint8_t>(),
    CB.data_ptr<c10::Half>(),
    Z.data_ptr<c10::Half>(),
    k,m,n
  );
}


__global__ void cuda_lookupmatmul_d4_k32_kernel(
    const c10::Half* __restrict__ X,      // k x n
    const uint8_t* __restrict__ YIs,      // m x (n/4)
    const c10::Half* __restrict__ CB,     // 256 x 4
    c10::Half* __restrict__ Z,            // k x m
    size_t K,
    size_t M,
    size_t N) {

  long m1 = blockIdx.x;
  long k1 = blockIdx.y;

  __shared__ c10::Half Y_cache[32*16];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;  
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;   
  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c0;               
  fill_fragment(c0, __float2half(0.0));

  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c1;    
  fill_fragment(c1, __float2half(0.0));

  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c2;    
  fill_fragment(c2, __float2half(0.0));

  wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c3;    
  fill_fragment(c3, __float2half(0.0));

  for (long jn = 0; jn < N / 16; jn++) {
    for (long r = 0; r < 4; r++) {
      uint8_t yidxs = *(uint8_t*)(YIs + jn*(4*M) + m1*4*32 + threadIdx.x*4 + r);
      ((uint64_t*)Y_cache)[threadIdx.x*4 + r] = ((uint64_t*)CB)[(yidxs & 255)];
    }

    load_matrix_sync(a, (const __half*)(X + 16*N*(2*k1+0) + 16*jn), N);

    load_matrix_sync(b, (const __half*)Y_cache, 16);
    mma_sync(c0, a, b, c0);
    
    load_matrix_sync(b, (const __half*)Y_cache + 16*16, 16);
    mma_sync(c1, a, b, c1);

    load_matrix_sync(a, (const __half*)(X + 16*N*(2*k1+1) + 16*jn), N);
    mma_sync(c3, a, b, c3);

    load_matrix_sync(b, (const __half*)Y_cache, 16);
    mma_sync(c2, a, b, c2);
  }
  
  store_matrix_sync((__half*)(&Z[16*M*(2*k1+0) + 32*m1 +  0]), c0, M, wmma::mem_row_major);
  store_matrix_sync((__half*)(&Z[16*M*(2*k1+0) + 32*m1 + 16]), c1, M, wmma::mem_row_major);
  store_matrix_sync((__half*)(&Z[16*M*(2*k1+1) + 32*m1 +  0]), c2, M, wmma::mem_row_major);
  store_matrix_sync((__half*)(&Z[16*M*(2*k1+1) + 32*m1 + 16]), c3, M, wmma::mem_row_major);
}


void lookupmatmul_d4_k32(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
) {
  auto k = X.sizes()[0];
  auto m = YIs.sizes()[0];
  auto n = X.sizes()[1];

  assert(X.dtype() == torch::kFloat16);
  assert(YIs.dtype() == torch::kUInt8);
  assert(CB.dtype() == torch::kFloat16);
  assert(Z.dtype() == torch::kFloat16);

  assert(Z.sizes()[0] == k);
  assert(YIs.sizes()[1] * 4 == n);
  assert(Z.sizes()[1] == m);

  assert(k % 16 == 0);
  assert(m % 32 == 0);
  assert(n % 16 == 0);

  const dim3 threads(32);
  const dim3 blocks(m/32,k/32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_lookupmatmul_d4_k32_kernel<<<blocks, threads, 0, stream>>>(
    X.data_ptr<c10::Half>(),
    YIs.data_ptr<uint8_t>(),
    CB.data_ptr<c10::Half>(),
    Z.data_ptr<c10::Half>(),
    k,m,n
  );
}

#define DECOMPRESS_D4_BLOCK_SIZE 256

__global__ void cuda_decompress_d4_origorder_kernel(
    const uint8_t* __restrict__ YIs,      // m x (n/4)
    const c10::Half* __restrict__ CB,           // 256 x 4
    c10::Half* __restrict__ Y             // m x n
) {
  const long i = threadIdx.x + DECOMPRESS_D4_BLOCK_SIZE * blockIdx.x;

  for(long r = 0; r < 4; r++) {
    uint8_t yidx = ((uint8_t*)YIs)[i*4 + r];
    ((uint64_t*)Y)[i*4 + r] = ((uint64_t*)CB)[yidx & 255];
  }
}


void decompress_d4_origorder(
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Y         // m x n
) {
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(CB.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 4 == n);
  assert(CB.sizes()[0] == 256);
  assert(CB.sizes()[1] == 4);

  const dim3 threads(DECOMPRESS_D4_BLOCK_SIZE);
  const dim3 blocks(m*n/(16*DECOMPRESS_D4_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_d4_origorder_kernel<<<blocks, threads, 0, stream>>>(
    YIs.data_ptr<uint8_t>(),
    CB.data_ptr<c10::Half>(),
    Y.data_ptr<c10::Half>()
  );
}


__global__ void cuda_decompress_d4_kernel(
    const uint8_t* __restrict__ YIs,      // m x (n/4)
    const c10::Half* __restrict__ CB,     // 256 x 4
    c10::Half* __restrict__ Y,            // m x n
    size_t M,
    size_t N
) {
  const long i = threadIdx.x + DECOMPRESS_D4_BLOCK_SIZE * blockIdx.x;

  const long j = (i % (N/16))*M + (i / (N/16));

  for(long r = 0; r < 4; r++) {
    uint8_t yidx = ((uint8_t*)YIs)[j*4 + r];
    ((uint64_t*)Y)[i*4 + r] = ((uint64_t*)CB)[yidx & 255];
  }
}


void decompress_d4(
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Y         // m x n
) {
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(CB.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 4 == n);
  assert(CB.sizes()[0] == 256);
  assert(CB.sizes()[1] == 4);

  const dim3 threads(DECOMPRESS_D4_BLOCK_SIZE);
  const dim3 blocks(m*n/(16*DECOMPRESS_D4_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_d4_kernel<<<blocks, threads, 0, stream>>>(
    YIs.data_ptr<uint8_t>(),
    CB.data_ptr<c10::Half>(),
    Y.data_ptr<c10::Half>(),
    m,n
  );
}




// This is a terrible kernel, only use this to not call the pytorch version

#define DECOMPRESS_HI4B1C_BLOCK_SIZE 128

__global__ void cuda_decompress_hi4b1c_packed_kernel(
    const int32_t* __restrict__ YIs,     // m x (n/8)
    const c10::Half* __restrict__ CB,     // 16 x 1
    c10::Half* __restrict__ Y             // m x n
) {
  const long i = threadIdx.x + DECOMPRESS_HI4B1C_BLOCK_SIZE * blockIdx.x;

  // 0 2 4 6 1 3 5 7
  uint32_t packed = YIs[i];
  Y[i*8 + 7] = CB[packed & 15];
  Y[i*8 + 5] = CB[(packed >> 4) & 15];
  Y[i*8 + 3] = CB[(packed >> 8) & 15];
  Y[i*8 + 1] = CB[(packed >> 12) & 15];
  Y[i*8 + 6] = CB[(packed >> 16) & 15];
  Y[i*8 + 4] = CB[(packed >> 20) & 15];
  Y[i*8 + 2] = CB[(packed >> 24) & 15];
  Y[i*8 + 0] = CB[(packed >> 28) & 15];
}


void decompress_hi4b1c_packed(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,
    torch::Tensor &Y         // m x n
) {
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 8 == n);

  assert(CB.sizes()[0] == 16);
  assert(CB.sizes()[1] == 1);

  
  const dim3 threads(DECOMPRESS_HI4B1C_BLOCK_SIZE);
  const dim3 blocks(m*n/(8*DECOMPRESS_HI4B1C_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_hi4b1c_packed_kernel<<<blocks, threads, 0, stream>>>(
    YIs.data_ptr<int32_t>(),
    CB.data_ptr<c10::Half>(),
    Y.data_ptr<c10::Half>()
  );
}


// This is a terrible kernel, only use this to not call the pytorch version

#define DECOMPRESS_HI3B1C_BLOCK_SIZE 128

__global__ void cuda_decompress_hi3b1c_packed_kernel(
    const int32_t* __restrict__ YIs,     // m x (n/8)
    const c10::Half* __restrict__ CB,     // 16 x 1
    c10::Half* __restrict__ Y             // m x n
) {
  const long i = threadIdx.x + DECOMPRESS_HI3B1C_BLOCK_SIZE * blockIdx.x;

  // 0 2 4 6 1 3 5 7
  uint32_t packed = YIs[i];
  Y[i*8 + 7] = CB[packed & 15];
  Y[i*8 + 5] = CB[(packed >> 4) & 15];
  Y[i*8 + 3] = CB[(packed >> 8) & 15];
  Y[i*8 + 1] = CB[(packed >> 12) & 15];
  Y[i*8 + 6] = CB[(packed >> 16) & 15];
  Y[i*8 + 4] = CB[(packed >> 20) & 15];
  Y[i*8 + 2] = CB[(packed >> 24) & 15];
  Y[i*8 + 0] = CB[(packed >> 28) & 15];
}


void decompress_hi3b1c_packed(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,
    torch::Tensor &Y         // m x n
) {
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 8 == n);

  assert(CB.sizes()[0] == 8);
  assert(CB.sizes()[1] == 1);

  
  const dim3 threads(DECOMPRESS_HI3B1C_BLOCK_SIZE);
  const dim3 blocks(m*n/(8*DECOMPRESS_HI3B1C_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_hi3b1c_packed_kernel<<<blocks, threads, 0, stream>>>(
    YIs.data_ptr<int32_t>(),
    CB.data_ptr<c10::Half>(),
    Y.data_ptr<c10::Half>()
  );
}

// This is a terrible kernel, only use this to not call the pytorch version

#define DECOMPRESS_HI2B1C_BLOCK_SIZE 128

__global__ void cuda_decompress_hi2b1c_packed_kernel(
    const int32_t* __restrict__ YIs,     // m x (n/8)
    const c10::Half* __restrict__ CB,     // 16 x 1
    c10::Half* __restrict__ Y             // m x n
) {
  const long i = threadIdx.x + DECOMPRESS_HI2B1C_BLOCK_SIZE * blockIdx.x;

  // 0 2 4 6 1 3 5 7
  uint32_t packed = YIs[i];
  Y[i*8 + 7] = CB[packed & 15];
  Y[i*8 + 5] = CB[(packed >> 4) & 15];
  Y[i*8 + 3] = CB[(packed >> 8) & 15];
  Y[i*8 + 1] = CB[(packed >> 12) & 15];
  Y[i*8 + 6] = CB[(packed >> 16) & 15];
  Y[i*8 + 4] = CB[(packed >> 20) & 15];
  Y[i*8 + 2] = CB[(packed >> 24) & 15];
  Y[i*8 + 0] = CB[(packed >> 28) & 15];
}


void decompress_hi2b1c_packed(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,
    torch::Tensor &Y         // m x n
) {
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 8 == n);

  assert(CB.sizes()[0] == 4);
  assert(CB.sizes()[1] == 1);

  
  const dim3 threads(DECOMPRESS_HI2B1C_BLOCK_SIZE);
  const dim3 blocks(m*n/(8*DECOMPRESS_HI2B1C_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_hi2b1c_packed_kernel<<<blocks, threads, 0, stream>>>(
    YIs.data_ptr<int32_t>(),
    CB.data_ptr<c10::Half>(),
    Y.data_ptr<c10::Half>()
  );
}



// This is a terrible kernel, only use this to not call the pytorch version

#define DECOMPRESS_E81B_BLOCK_SIZE 4

__global__ void cuda_decompress_e81b_packed_kernel(
    const int64_t* __restrict__ YIs,     // m x (n/8)
    const c10::Half* __restrict__ CB,     // 256 x 8
    c10::Half* __restrict__ Y             // m x n
) {
  const long i = threadIdx.x + DECOMPRESS_E81B_BLOCK_SIZE * blockIdx.x;

  uint64_t packed = YIs[i];
  
#pragma unroll
  for (long j = 0; j < 8; j++) {
    uint64_t yidx = packed & 255;
    ((uint64_t*)Y)[(i*8 + j)*2] = ((uint64_t*)CB)[yidx*2];
    ((uint64_t*)Y)[(i*8 + j)*2 + 1] = ((uint64_t*)CB)[yidx*2 + 1];
    packed = packed >> 8;
  }
  
}

void decompress_e81b_packed(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,
    torch::Tensor &Y         // m x n
) {
  size_t m = Y.sizes()[0];
  size_t n = Y.sizes()[1];

  assert(YIs.is_contiguous());
  assert(Y.is_contiguous());

  assert(YIs.sizes()[0] == m);
  assert(YIs.sizes()[1] * 64 == n);

  assert(CB.sizes()[0] == 256);
  assert(CB.sizes()[1] == 8);

  at::DeviceGuard guard(CB.device());
  const dim3 threads(DECOMPRESS_E81B_BLOCK_SIZE);
  const dim3 blocks(m*n/(64*DECOMPRESS_E81B_BLOCK_SIZE));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_decompress_e81b_packed_kernel<<<blocks, threads, 0, stream>>>(
    YIs.data_ptr<int64_t>(),
    CB.data_ptr<c10::Half>(),
    Y.data_ptr<c10::Half>()
  );
}



__global__ void cuda_lookupmatmul_e81b_k8_kernel(
    const c10::Half* __restrict__ X,      // k x n
    const int64_t* __restrict__ YIs,      // m x (n/64)
    const c10::Half* __restrict__ CB,     // 256 x 8
    float* __restrict__ Z,
    size_t K,
    size_t M,
    size_t N) {

  long m1 = blockIdx.x;
  long k1 = blockIdx.y;

  __shared__ c10::Half Y_cache0[32*16];
  wmma::fragment<wmma::matrix_a, 8, 32, 16, __half, wmma::row_major> a0;  // 8 x 16
  wmma::fragment<wmma::matrix_b, 8, 32, 16, __half, wmma::col_major> b0;  // 32 x 16

  __shared__ c10::Half Y_cache1[32*16];
  wmma::fragment<wmma::matrix_a, 8, 32, 16, __half, wmma::row_major> a1;  // 8 x 16
  wmma::fragment<wmma::matrix_b, 8, 32, 16, __half, wmma::col_major> b1;  // 32 x 16
  
  wmma::fragment<wmma::accumulator, 8, 32, 16, float> c;                // 8 x 32
  fill_fragment(c, 0.0);


#pragma unroll
  for (long jn = 0; jn < N / 32; jn++) {
    uint32_t packed = ((uint32_t*)YIs)[(m1*32 + threadIdx.x)*(N/32) + jn];
#pragma unroll
    for (long r = 0; r < 2; r++) {
      uint32_t yidx = packed & 255;
      ((uint64_t*)Y_cache0)[(threadIdx.x*2 + r)*2] = ((uint64_t*)CB)[yidx*2];
      ((uint64_t*)Y_cache0)[(threadIdx.x*2 + r)*2 + 1] = ((uint64_t*)CB)[yidx*2 + 1];
      packed = packed >> 8;
    }
#pragma unroll
    for (long r = 0; r < 2; r++) {
      uint32_t yidx = packed & 255;
      ((uint64_t*)Y_cache1)[(threadIdx.x*2 + r)*2] = ((uint64_t*)CB)[yidx*2];
      ((uint64_t*)Y_cache1)[(threadIdx.x*2 + r)*2 + 1] = ((uint64_t*)CB)[yidx*2 + 1];
      packed = packed >> 8;
    }

    load_matrix_sync(a0, (const __half*)(X + 8*N*k1 + 32*jn), N);
    load_matrix_sync(b0, (const __half*)Y_cache0, 16);
    mma_sync(c, a0, b0, c);
    
    load_matrix_sync(a1, (const __half*)(X + 8*N*k1 + 32*jn + 16), N);
    load_matrix_sync(b1, (const __half*)Y_cache1, 16);
    mma_sync(c, a1, b1, c);

  }
  
  store_matrix_sync(&Z[8*M*k1 + 32*m1], c, M, wmma::mem_row_major);
}


void lookupmatmul_e81b_k8(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/64)
    torch::Tensor CB,       // 256 x 8
    torch::Tensor Z         // k x m
) {
  auto k = X.sizes()[0];
  auto m = YIs.sizes()[0];
  auto n = X.sizes()[1];

  assert(Z.sizes()[0] == k);
  assert(YIs.sizes()[1] * 64 == n);
  assert(Z.sizes()[1] == m);

  assert(k <= 8);
  assert(m % 32 == 0);
  assert(n % 32 == 0);
  
  at::DeviceGuard guard(CB.device());
  const dim3 threads(32);
  const dim3 blocks(m/32, k/8);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cuda_lookupmatmul_e81b_k8_kernel<<<blocks, threads, 0, stream>>>(
    X.data_ptr<c10::Half>(),
    YIs.data_ptr<int64_t>(),
    CB.data_ptr<c10::Half>(),
    Z.data_ptr<float>(),
    k,m,n
  );
}
