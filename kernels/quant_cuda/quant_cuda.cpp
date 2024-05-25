#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

// standard forward operations

void vecquant2matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
); 

void vecquant2matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant3matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
); 

void vecquant3matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant4matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
); 

void vecquant4matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant8matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
); 

void vecquant8matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}

// methods based on reconstruction (unpacking)

void vecquant4recons_v1_cuda(
  torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros
);

void vecquant4recons_v1(
  torch::Tensor mat, torch::Tensor res,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  vecquant4recons_v1_cuda(mat, res, scales, zeros);
}

void vecquant4recons_v2_cuda(
  torch::Tensor mat, torch::Tensor res,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant4recons_v2(
  torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros, torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  vecquant4recons_v2_cuda(mat, res, scales, zeros, g_idx);
}

void vecquant2recons_v2_cuda(
  torch::Tensor mat, torch::Tensor res,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant2recons_v2(
  torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros, torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  vecquant2recons_v2_cuda(mat, res, scales, zeros, g_idx);
}

void vecquant4matmul_v1_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant4matmul_v1_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_v1_faster_cuda(vec, mat, mul, scales, zeros);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant2matmul", &vecquant2matmul, "Vector 2-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant8matmul", &vecquant8matmul, "Vector 8-bit Quantized Matrix Multiplication (CUDA)");

  // Reconstruction Kernel
  m.def("vecquant4recons_v1", &vecquant4recons_v1, "Vector 4-bit Quantized Matrix Reconstruction (CUDA)");
  m.def("vecquant4recons_v2", &vecquant4recons_v2, "Vector 4-bit Quantized Matrix Reconstruction (CUDA) with group-size support");
  m.def("vecquant2recons_v2", &vecquant2recons_v2, "Vector 2-bit Quantized Matrix Reconstruction (CUDA) with group-size support");
}