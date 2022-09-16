/*
   CUDA extension for SAPA
   Modified from CARAFE https://github.com/myownskyW7/CARAFE
*/
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int atn_forward_cuda(at::Tensor attn_data, at::Tensor value_data,
                              int kernel_size, int scale_factor,
                              at::Tensor output);

int atn_backward_cuda(at::Tensor top_grad, at::Tensor attn_data,
                               at::Tensor value_data, int kernel_size,
                               int scale_factor,
                               at::Tensor attn_grad, at::Tensor value_grad);
#endif

int atn_forward(at::Tensor attn_data, at::Tensor value_data,
                         int kernel_size, int scale_factor,
                         at::Tensor output) {
  if (attn_data.device().is_cuda()) {
#ifdef WITH_CUDA
    return atn_forward_cuda(attn_data, value_data, kernel_size,
        scale_factor, output);
#else
    AT_ERROR("atn is not compiled with GPU support");
#endif
  }
  AT_ERROR("atn is not implemented on CPU");
}

int atn_backward(at::Tensor top_grad, at::Tensor attn_data,
                               at::Tensor value_data, int kernel_size,
                               int scale_factor,
                               at::Tensor attn_grad, at::Tensor value_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    return atn_backward_cuda(top_grad, attn_data, value_data, kernel_size,
        scale_factor, attn_grad, value_grad);
#else
    AT_ERROR("atn is not compiled with GPU support");
#endif
  }
  AT_ERROR("atn is not implemented on CPU");

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &atn_forward, "atn_forward");
  m.def("backward", &atn_backward, "atn_backward");
}