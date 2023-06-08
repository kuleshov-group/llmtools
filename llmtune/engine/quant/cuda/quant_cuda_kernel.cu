#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(
    double* address,
    double val
) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 700
// adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int *address_as_ui = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short hsum = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        hsum += val;
        old = reinterpret_cast<size_t>(address) & 2
                 ? (old & 0xffff) | (hsum << 16)
                 : (old & 0xffff0000) | hsum;
        old = atomicCAS(address_as_ui, assumed, old);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}
#endif
#endif

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

template <typename scalar_t>
__global__ void VecQuant2MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const     int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const   int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const   int* __restrict__ zeros,
    const   int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width
);

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const   int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

__global__ void VecQuant2MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

__global__ void VecQuant3MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

template <typename scalar_t>
__global__ void VecQuant4MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const    int* __restrict__ zeros,
    const    int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width
);

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT2 =  16;
const int BLOCKHEIGHT3 =  24;
const int BLOCKHEIGHT4 =  32;
const int BLOCKHEIGHT8 =  64;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

void vecquant2matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant2matmul_cuda", ([&] {
      VecQuant2MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant2MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const   int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT2 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 16;
  int k = 0;

  int z_w = w / 16;
  int z_mod = (w % 16) * 2;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x3) + 1);

    res += (scale * scalar_t((tmp >> 0) & 0x3) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 2) & 0x3) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 4) & 0x3) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 6) & 0x3) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp >> 8) & 0x3) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp >> 10) & 0x3) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp >> 12) & 0x3) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp >> 14) & 0x3) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp >> 16) & 0x3) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp >> 18) & 0x3) - zero) * blockvec[k + 9];
    res += (scale * scalar_t((tmp >> 20) & 0x3) - zero) * blockvec[k + 10];
    res += (scale * scalar_t((tmp >> 22) & 0x3) - zero) * blockvec[k + 11];
    res += (scale * scalar_t((tmp >> 24) & 0x3) - zero) * blockvec[k + 12];
    res += (scale * scalar_t((tmp >> 26) & 0x3) - zero) * blockvec[k + 13];
    res += (scale * scalar_t((tmp >> 28) & 0x3) - zero) * blockvec[k + 14];
    res += (scale * scalar_t((tmp >> 30) & 0x3) - zero) * blockvec[k + 15];

    i += width;
    k += 16;
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant3matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant3matmul_cuda", ([&] {
      VecQuant3MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const   int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = 0;

  int z_w = (w / 32) * 3;
  int z_mod = w % 32;
  int z_bit;

  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;
  unsigned int z_tmp;

  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = scale * scalar_t((z_tmp) + 1);
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = scale * scalar_t((z_tmp) + 1);
    } else {
      zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1);
    }

    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;

    res += (scale * scalar_t((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp2 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp2 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];

    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;

    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];

    i += width;
    k += 10;
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant4matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_cuda", ([&] {
      VecQuant4MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const       int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    int tmp_k = 0;
    scalar_t scales_tmp[8];
    scalar_t zeros_tmp[8];
    while (tmp_k < 8) {
      int g = g_idx[g_h + k + tmp_k];
      scalar_t scale = scales[g * width + w];
      scalar_t zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1);
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
      tmp_k += 1;
    }

    res += (scales_tmp[0] * scalar_t((tmp >> 0) & 0xF) - zeros_tmp[0]) * blockvec[k + 0];
    res += (scales_tmp[1] * scalar_t((tmp >> 4) & 0xF) - zeros_tmp[1]) * blockvec[k + 1];
    res += (scales_tmp[2] * scalar_t((tmp >> 8) & 0xF) - zeros_tmp[2]) * blockvec[k + 2];
    res += (scales_tmp[3] * scalar_t((tmp >> 12) & 0xF) - zeros_tmp[3]) * blockvec[k + 3];
    res += (scales_tmp[4] * scalar_t((tmp >> 16) & 0xF) - zeros_tmp[4]) * blockvec[k + 4];
    res += (scales_tmp[5] * scalar_t((tmp >> 20) & 0xF) - zeros_tmp[5]) * blockvec[k + 5];
    res += (scales_tmp[6] * scalar_t((tmp >> 24) & 0xF) - zeros_tmp[6]) * blockvec[k + 6];
    res += (scales_tmp[7] * scalar_t((tmp >> 28) & 0xF) - zeros_tmp[7]) * blockvec[k + 7];

    i += width;
    k += 8;
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant8matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT8 - 1) / BLOCKHEIGHT8,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_cuda", ([&] {
      VecQuant8MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const   int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT8 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 4;
  int k = 0;

  int z_w = w / 4;
  int z_mod = (w % 4) * 8;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1);

    res += (scale * scalar_t((tmp >> 0) & 0xFF) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 8) & 0xFF) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 16) & 0xFF) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 24) & 0xFF) - zero) * blockvec[k + 3];

    i += width;
    k += 4;
  }

  atomicAdd(&mul[b * width + w], res);
}


void vecquant2matmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant2MatMulKernelFaster<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<int>(),
    batch, vec_height, height, width, zero_width, groupsize
  );
}

__global__ void VecQuant2MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
  int batch,
  int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT2 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[16][16];
  int val = threadIdx.x / 16;
  int off = threadIdx.x % 16;
  for (; val < 16; val += BLOCKWIDTH / 16) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0x3), __int2half_rn(val >> 2)
    );
  }

  int i = width * h + w;
  int g_h = h * 16;
  int k = 0;

  int z_w = w / 16;
  int z_mod = (w % 16) * 2;

  float res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
  float scale_f = scales[g * width + w];
    half2 scale = __float2half2_rn(scale_f);
    half2 zero = __float2half2_rn(-(scale_f * (((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0x3) + 1)));

    res2 = {};
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xf][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  4) & 0xf][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xf][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 12) & 0xf][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xf][off], scale, zero), blockvec[k + 4], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 20) & 0xf][off], scale, zero), blockvec[k + 5], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xf][off], scale, zero), blockvec[k + 6], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 28) & 0xf][off], scale, zero), blockvec[k + 7], res2);
  i += width;
    k += 8;
    res += __half2float(res2.x) + __half2float(res2.y);
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant3matmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelFaster<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<int>(),
    batch, vec_height, height, width, zero_width, groupsize
  );
}

__global__ void VecQuant3MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
  int batch,
  int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0x7), __int2half_rn(val >> 3)
    );
  }

  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = 0;

  int z_w = (w / 32) * 3;
  int z_mod = w % 32;
  int z_bit;

  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }

  float res = 0;
  half2 res2;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;
  unsigned int z_tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
  float scale_f = scales[g * width + w];
    half2 scale = __float2half2_rn(scale_f);
    half2 zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = __float2half2_rn(-(scale_f * ((z_tmp) + 1)));
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = __float2half2_rn(-(scale_f * ((z_tmp) + 1)));
    } else {
      zero = __float2half2_rn(-(scale_f * (((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1)));
    }

    res2 = {};
    tmp1 = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 5], res2);
    tmp2 >>= 4;
    k += 6;
    res2 = __hfma2(__hfma2(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 4], res2);
    tmp1 >>= 2;
    k += 5;
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
    i += width;
    k += 5;
    res += __half2float(res2.x) + __half2float(res2.y);
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant4matmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_SWITCH(vec.type(), "vecquant4matmul_faster_cuda",
    AT_DISPATCH_CASE(at::ScalarType::Half, ([&] {
      VecQuant4MatMulKernelFaster<<<blocks, threads>>>(
        (half2*) vec.data_ptr<scalar_t>(),
        mat.data_ptr<int>(),
        mul.data_ptr<scalar_t>(),
        scales.data_ptr<scalar_t>(),
        zeros.data_ptr<int>(),
        g_idx.data_ptr<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  ));
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const    int* __restrict__ zeros,
    const    int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0xF), __int2half_rn(val >> 4)
    );
  }

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  scalar_t res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = as_int(g_idx[g_h + (k * 2)]);
    scalar_t scale_f = scales[g * width + w];
    half2 scale = __half2half2(scale_f);
    half2 zero = __half2half2(__hmul(-scale_f, __int2half_rn(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1)));

    res2 = {};
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xff][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xff][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off], scale, zero), blockvec[k + 3], res2);
    i += width;
    k += 4;
    res = __hadd(res, __hadd(res2.x, res2.y));;
  }

  __half* mul2 = (__half*)mul;
  atomicAdd(&mul2[b * width + w], res);
}

template <typename scalar_t>
__global__ void VecQuant4ReconsV1Kernel(
    const       int* __restrict__ mat,
           scalar_t* __restrict__ res,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int n_rows = h * 8 + b;
  int n_cols = w;
  scalar_t scale = scales[w];
  scalar_t zero = zeros[w];
  int i = width * h + width * (b / 8) + w;
  int shift = b % 8 * 4;
  unsigned int tmp = as_unsigned(mat[i]);
  scalar_t result = (scale * scalar_t((tmp >> shift) & 0xF) - zero);
  res[n_rows * width + n_cols] = result;
}

void vecquant4recons_v1_cuda(
  torch::Tensor mat,
  torch::Tensor res,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = BLOCKWIDTH;
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    scales.type(), "vecquant4recons_v1_cuda", ([&] {
      VecQuant4ReconsV1Kernel<<<blocks, threads>>>(
        mat.data<int>(), res.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<scalar_t>(),
        height, width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant4ReconsV2Kernel(
    const       int* __restrict__ mat,
           scalar_t* __restrict__ res,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const       int* __restrict__ g_idx,
    int height,
    int width,
    int zero_width
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int n_rows = h * 8 + b;
  int n_cols = w;
  int z_rows = as_int(g_idx[n_rows]);
  int z_cols = n_cols / 8;
  int z_shift = (n_cols % 8) * 4;
  scalar_t scale = scales[z_rows * width + n_cols];
  scalar_t zero = scale * scalar_t(((as_unsigned(zeros[z_rows * zero_width + z_cols]) >> z_shift) & 0xF) + 1);
  int i = width * h + width * (b / 8) + w;
  int shift = b % 8 * 4;
  unsigned int tmp = as_unsigned(mat[i]);
  scalar_t result = (scale * scalar_t((tmp >> shift) & 0xF) - zero);
  res[n_rows * width + n_cols] = result;
}

void vecquant4recons_v2_cuda(
  torch::Tensor mat,
  torch::Tensor res,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = BLOCKWIDTH;
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    scales.type(), "vecquant4recons_v2_cuda", ([&] {
      VecQuant4ReconsV2Kernel<<<blocks, threads>>>(
        mat.data<int>(), res.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        g_idx.data<int>(), height, width, zero_width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant2ReconsV2Kernel(
    const       int* __restrict__ mat,
           scalar_t* __restrict__ res,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const       int* __restrict__ g_idx,
    int height,
    int width,
    int zero_width
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT2 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int n_rows = h * 16 + b;
  int n_cols = w;
  int z_rows = as_int(g_idx[n_rows]);
  int z_cols = n_cols / 16;
  int z_shift = (n_cols % 16) * 2;
  scalar_t scale = scales[z_rows * width + n_cols];
  scalar_t zero = scale * scalar_t(((as_unsigned(zeros[z_rows * zero_width + z_cols]) >> z_shift) & 0x3) + 1);
  int i = width * h + width * (b / 16) + w;
  int shift = b % 16 * 2;
  unsigned int tmp = as_unsigned(mat[i]);
  scalar_t result = (scale * scalar_t((tmp >> shift) & 0x3) - zero);
  res[n_rows * width + n_cols] = result;
}

void vecquant2recons_v2_cuda(
  torch::Tensor mat,
  torch::Tensor res,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = BLOCKWIDTH;
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    scales.type(), "vecquant2recons_v2_cuda", ([&] {
      VecQuant2ReconsV2Kernel<<<blocks, threads>>>(
        mat.data<int>(), res.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        g_idx.data<int>(), height, width, zero_width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant4MatMulV1KernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0xF), __int2half_rn(val >> 4)
    );
  }

  int i = width * h + w;
  int k = 0;

  scalar_t res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    scalar_t scale_f = scales[w];
    scalar_t zero_f = zeros[w];
    half2 scale = __half2half2(scale_f);
    half2 zero = __half2half2(-zero_f);

    res2 = {};
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xff][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xff][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off], scale, zero), blockvec[k + 3], res2);
    i += width;
    k += 4;
    res = __hadd(res, __hadd(res2.x, res2.y));
  }

  __half* mul2 = (__half*)mul;
  atomicAdd(&mul2[b * width + w], res);
}

void vecquant4matmul_v1_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1) / 2;
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_SWITCH(vec.type(), "vecquant4matmul_v1_faster_cuda",
    AT_DISPATCH_CASE(at::ScalarType::Half, ([&] {
      VecQuant4MatMulV1KernelFaster<<<blocks, threads>>>(
        (half2*) vec.data_ptr<scalar_t>(),
        mat.data_ptr<int>(),
        mul.data_ptr<scalar_t>(),
        scales.data_ptr<scalar_t>(),
        zeros.data_ptr<scalar_t>(),
        batch, vec_height, height, width
      );
    })
  ));
}

#define GROUPS_SEQ_V2 32
#define BLOCKWIDTH_SEQ_V2 64
#define BLOCKHEIGHT4_SEQ_V2 32

template <typename scalar_t>
__global__ void VecQuant4MatMulSeqV2Kernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat_t,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales_t,
    const       int* __restrict__ zeros_t,
    const       int* __restrict__ g_idx,
    int batch,
    int vec_width,
    int height,
    int width,
    int zero_width
) {
  int b = blockIdx.z;
  int mat_t_n_row = BLOCKWIDTH_SEQ_V2 * blockIdx.x + threadIdx.x;
  int mat_t_n_col = BLOCKHEIGHT4_SEQ_V2 * blockIdx.y; // BLOCKHEIGHT4 is BLOCKWIDTH4 here

  __shared__ scalar_t blockvec[BLOCKWIDTH_SEQ_V2];
  __shared__ int g_idx_ptr[BLOCKWIDTH_SEQ_V2];
  __shared__ int zero_loc[GROUPS_SEQ_V2];
  __shared__ int zero_shifter[GROUPS_SEQ_V2];
  blockvec[threadIdx.x] = vec[b * vec_width + mat_t_n_row];
  g_idx_ptr[threadIdx.x] = g_idx[mat_t_n_col * 8 + threadIdx.x];
  if (threadIdx.x < GROUPS_SEQ_V2) {
    zero_loc[threadIdx.x] = threadIdx.x / 8;
  }
  else{
    int idx = threadIdx.x - GROUPS_SEQ_V2;
    zero_shifter[idx] = idx % 8 * 4;
  }
  __syncthreads();
  
  scalar_t res = 0;
  unsigned int tmp;
  int k = 0;
  int mat_i = mat_t_n_row * width + mat_t_n_col;
  int zeros_i_base = mat_t_n_row * zero_width;
  int scales_i_base = mat_t_n_row * GROUPS_SEQ_V2;

  while (k < BLOCKWIDTH_SEQ_V2) {
    tmp = as_unsigned(mat_t[mat_i]);
    
    scalar_t scales_tmp[8];
    scalar_t zeros_tmp[8];
    int g;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      g = g_idx_ptr[k + i];
      scalar_t scale = scales_t[scales_i_base + g];
      scalar_t zero = scale * scalar_t(((as_unsigned(zeros_t[zeros_i_base + zero_loc[g]]) >> zero_shifter[g]) & 0xF) + 1);
      scales_tmp[i] = scale;
      zeros_tmp[i] = zero;
    }

    res += (scales_tmp[0] * scalar_t((tmp >> 0) & 0xF) - zeros_tmp[0]) * blockvec[k + 0];
    res += (scales_tmp[1] * scalar_t((tmp >> 4) & 0xF) - zeros_tmp[1]) * blockvec[k + 1];
    res += (scales_tmp[2] * scalar_t((tmp >> 8) & 0xF) - zeros_tmp[2]) * blockvec[k + 2];
    res += (scales_tmp[3] * scalar_t((tmp >> 12) & 0xF) - zeros_tmp[3]) * blockvec[k + 3];
    res += (scales_tmp[4] * scalar_t((tmp >> 16) & 0xF) - zeros_tmp[4]) * blockvec[k + 4];
    res += (scales_tmp[5] * scalar_t((tmp >> 20) & 0xF) - zeros_tmp[5]) * blockvec[k + 5];
    res += (scales_tmp[6] * scalar_t((tmp >> 24) & 0xF) - zeros_tmp[6]) * blockvec[k + 6];
    res += (scales_tmp[7] * scalar_t((tmp >> 28) & 0xF) - zeros_tmp[7]) * blockvec[k + 7];

    k += 8;
    mat_i += 1;
  }

  atomicAdd(&mul[b * height + mat_t_n_row], res);
}


void vecquant4matmul_seq_v2_cuda(
  torch::Tensor vec,
  torch::Tensor mat_t,
  torch::Tensor mul,
  torch::Tensor scales_t,
  torch::Tensor zeros_t,
  torch::Tensor g_idx
) {
  // mat (n / 8, m)
  // mat_t (m, n / 8)
  // zeros need repack
  // zeros (groups, m / 8)
  // zeros_t should be (m, groups / 8)
  // scales (groups, m)
  // scales_t (m, groups)
  int batch = vec.size(0);
  int vec_width = vec.size(1);
  int height = mat_t.size(0);
  int width = mat_t.size(1);
  int zero_width = zeros_t.size(1);

  dim3 blocks(
    (height + BLOCKWIDTH_SEQ_V2 - 1) / BLOCKWIDTH_SEQ_V2,
    (width + BLOCKHEIGHT4_SEQ_V2 - 1) / BLOCKHEIGHT4_SEQ_V2,
    batch
  );
  dim3 threads(BLOCKWIDTH_SEQ_V2);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_seq_v2_cuda", ([&] {
      VecQuant4MatMulSeqV2Kernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat_t.data<int>(), mul.data<scalar_t>(),
        scales_t.data<scalar_t>(), zeros_t.data<int>(), g_idx.data<int>(),
        batch, vec_width, height, width, zero_width
      );
    })
  );
}
