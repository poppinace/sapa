/*
   CUDA extension for SAPA
   Modified from CARAFE https://github.com/myownskyW7/CARAFE
*/
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int qk_forward_cuda(at::Tensor query_data, at::Tensor key_data,
                              int kernel_size, int scale_factor,
                              at::Tensor output);

int qk_backward_cuda(at::Tensor top_grad, at::Tensor query_data,
                               at::Tensor key_data, int kernel_size,
                               int scale_factor,
                               at::Tensor query_grad, at::Tensor key_grad);
#endif

int qk_forward(at::Tensor query_data, at::Tensor key_data,
                         int kernel_size, int scale_factor,
                         at::Tensor output) {
  if (query_data.device().is_cuda()) {
#ifdef WITH_CUDA
    return qk_forward_cuda(query_data, key_data, kernel_size,
        scale_factor, output);
#else
    AT_ERROR("qk is not compiled with GPU support");
#endif
  }
  AT_ERROR("qk is not implemented on CPU");
}

int qk_backward(at::Tensor top_grad, at::Tensor query_data,
                               at::Tensor key_data, int kernel_size,
                               int scale_factor,
                               at::Tensor query_grad, at::Tensor key_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return qk_backward_cuda(top_grad, query_data, key_data, kernel_size,
    scale_factor, query_grad, key_grad);
#else
    AT_ERROR("qk is not compiled with GPU support");
#endif
  }
  AT_ERROR("qk is not implemented on CPU");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &qk_forward, "qk_forward");
  m.def("backward", &qk_backward, "qk_backward");
}
