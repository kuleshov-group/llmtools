#include <torch/extension.h>

#include <iostream>
#include <cassert>

void lookupmatmul_d4_k8(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
);

void lookupmatmul_d4_k16(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
);

void lookupmatmul_d4_k32(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
);

void decompress_d4(
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Y         // m x n
);

void decompress_d4_origorder(
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Y         // m x n
);

torch::Tensor decompress_packed_e8p(
    torch::Tensor weights_compressed,      // m x (n/8)
    torch::Tensor codebook_abs       // 256 x 8
);

torch::Tensor decode_matvec_e8p(
    torch::Tensor x,
    torch::Tensor weights_compressed,
    torch::Tensor codebook_abs
);

void decompress_hi4b1c_packed(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,       // 16 x 1
    torch::Tensor &Y        // m x n
);

void decompress_hi3b1c_packed(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,       // 16 x 1
    torch::Tensor &Y        // m x n
);

void decompress_hi2b1c_packed(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,       // 16 x 1
    torch::Tensor &Y        // m x n
);

void decompress_e81b_packed(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,       // 256 x 8
    torch::Tensor &Y        // m x n
);

void lookupmatmul_e81b_k8(
    torch::Tensor X,        // k x n
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Z         // k x m
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lookupmatmul_d4_k8", &lookupmatmul_d4_k8, "lookupmatmul_d4_k8");
  m.def("lookupmatmul_d4_k16", &lookupmatmul_d4_k16, "lookupmatmul_d4_k16");
  m.def("lookupmatmul_d4_k32", &lookupmatmul_d4_k32, "lookupmatmul_d4_k32");
  m.def("decompress_d4", &decompress_d4, "decompress_d4");
  m.def("decompress_d4_origorder", &decompress_d4_origorder, "decompress_d4_origorder");
  m.def("decompress_packed_e8p", &decompress_packed_e8p, "decompress_packed_e8p");
  m.def("decode_matvec_e8p", &decode_matvec_e8p, "decode_matvec_e8p");
  m.def("decompress_hi4b1c_packed", &decompress_hi4b1c_packed, "decompress_hi4b1c_packed");
  m.def("decompress_hi3b1c_packed", &decompress_hi3b1c_packed, "decompress_hi3b1c_packed");
  m.def("decompress_hi2b1c_packed", &decompress_hi2b1c_packed, "decompress_hi2b1c_packed");
  m.def("decompress_e81b_packed", &decompress_e81b_packed, "decompress_e81b_packed");
  m.def("lookupmatmul_e81b_k8", &lookupmatmul_e81b_k8, "lookupmatmul_e81b_k8");
}
