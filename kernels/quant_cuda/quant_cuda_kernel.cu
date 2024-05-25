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

template <typename scalar_t>
__global__ void VecQuant2MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const     int* __restrict__ zeros,
  const       int* __restrict__ g_idx,
    int batch,
    int vec_height,   
    int height,
    int width,
  int zero_width
);

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const     int* __restrict__ zeros,
  const       int* __restrict__ g_idx,
    int batch,
    int vec_height,   
    int height,
    int width,
  int zero_width
);

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const     int* __restrict__ zeros,
  const       int* __restrict__ g_idx,
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
    const     int* __restrict__ zeros,
  const       int* __restrict__ g_idx,
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

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}


void vecquant2matmul_cuda(
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
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant2matmul_cuda", ([&] {
      VecQuant2MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(), 
        batch, vec_height, height, width, zero_width
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
    const     int* __restrict__ zeros,
    const     int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
  int zero_width
) {
  int h = BLOCKHEIGHT2 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  
  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int g_h = h * 16;
  int k;
  unsigned int g;
  scalar_t w_tmp;
  
  int z_w = w / 16; 
  int z_mod = (w % 16) * 2;
  
  float weight[BLOCKWIDTH];
  
  for (k = 0; k <  BLOCKWIDTH; ++k){  
  int k_w = (k / 16); 
  int k_bit = (k % 16) * 2;
  
    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scalar_t((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x3) + 1);
  
    w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3);
    
  weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){  
  res = 0;
  
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();
  for (k = 0; k <  BLOCKWIDTH; ++k){  
    res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
}

void vecquant3matmul_cuda(
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
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant3matmul_cuda", ([&] {
      VecQuant3MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(), 
        batch, vec_height, height, width, zero_width
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
    const       int* __restrict__ zeros,
    const     int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
  int zero_width
) {
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  
  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k;
  unsigned int g;
  scalar_t w_tmp;
  
  int z_w = (w / 32) * 3; 
  int z_mod = w % 32;
  int z_bit;
  unsigned int z_tmp;
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
  
  float weight[BLOCKWIDTH];
  
  for (k = 0; k <  BLOCKWIDTH; ++k){  
  int k_w = (k / 32) * 3; 
  int k_mod = k % 32;
  int k_bit;
    
  if (k_mod != 10){
    if (k_mod != 21){
        k_bit = k_mod;
        if (k_bit > 21){
      k_bit -= 22;
      k_bit *= 3;
      k_bit += 2;
      k_w += 2;
        } else if (k_bit > 10){
      k_bit -= 11;
      k_bit *= 3;
      k_bit += 1;
      k_w += 1;
        } else {
      k_bit *= 3;
        }
    } else {
        k_w += 1;
    }
  }
  
    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = scalar_t((z_tmp) + 1);
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = scalar_t((z_tmp) + 1);
    } else {
      zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1);
    }
  
    if (k_mod == 10) {
      w_tmp = (as_unsigned(mat[i + (k_w * width)]) >> 30) | ((as_unsigned(mat[i + ((k_w + 1)* width)]) << 2) & 0x4);
    } else if (k_mod == 21){
      w_tmp = (as_unsigned(mat[i + (k_w * width)]) >> 31) | ((as_unsigned(mat[i + ((k_w + 1)* width)]) << 1) & 0x6);
    } else {
      w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x7);
    }
  weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){  
  res = 0;
  
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();
  for (k = 0; k <  BLOCKWIDTH; ++k){  
    res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
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
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
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
    const     int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
  int zero_width
) {
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  
  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int g_h = h * 8;
  int k;
  unsigned int g;
  scalar_t w_tmp;
  

  int z_w = w / 8; 
  int z_mod = (w % 8) * 4;
  
  float weight[BLOCKWIDTH];
  
  for (k = 0; k <  BLOCKWIDTH; ++k){  
  int k_w = (k / 8); 
  int k_bit = (k % 8) * 4;
  
    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1);
  
    w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF);
    
  weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){  
  res = 0;
  
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();
  for (k = 0; k <  BLOCKWIDTH; ++k){  
    res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
}

void vecquant8matmul_cuda(
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
    (height + BLOCKHEIGHT8 - 1) / BLOCKHEIGHT8,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_cuda", ([&] {
      VecQuant8MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(), 
        batch, vec_height, height, width, zero_width
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
    const       int* __restrict__ zeros,
    const     int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
  int zero_width
) {
  int h = BLOCKHEIGHT8 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  
  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int g_h = h * 4;
  int k;
  unsigned int g;
  scalar_t w_tmp;
  
  int z_w = w / 4; 
  int z_mod = (w % 4) * 8;
  
  float weight[BLOCKWIDTH];
  
  for (k = 0; k <  BLOCKWIDTH; ++k){  
  int k_w = (k / 4); 
  int k_bit = (k % 4) * 8;
  
    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1);
  
    w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF);
    
  weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){  
  res = 0;
  
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();
  for (k = 0; k <  BLOCKWIDTH; ++k){  
    res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
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
