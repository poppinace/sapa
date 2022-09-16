/*
   CUDA extension for SAPA
   Modified from CARAFE https://github.com/myownskyW7/CARAFE
*/
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

int ATNForwardLauncher(const at::Tensor attn_data,
                       const at::Tensor value_data,
                       const int kernel_size,
                       const int scale_factor,
                       const int batch_size,
                       const int channels,
                       const int height,
                       const int width,
                       at::Tensor output);

int ATNBackwardLauncher(const at::Tensor top_grad,
                        const at::Tensor attn_data,
                        const at::Tensor value_data,
                        const int kernel_size,
                        const int scale_factor,
                        const int batch_size,
                        const int channels,
                        const int height,
                        const int width,
                        at::Tensor attn_grad,
                        at::Tensor value_grad);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int atn_forward_cuda(at::Tensor attn_data, at::Tensor value_data,
                              int kernel_size, int scale_factor,
                              at::Tensor output)
{
  CHECK_INPUT(attn_data);
  CHECK_INPUT(value_data);
  CHECK_INPUT(output);
  at::DeviceGuard guard(attn_data.device());

  int batch_size = output.size(0);
  int data_height = output.size(1);
  int data_width = output.size(2);
  int num_channels = output.size(3);

  ATNForwardLauncher(attn_data, value_data, kernel_size,
                     scale_factor, batch_size, num_channels, data_height,
                     data_width, output);

  return 1;
}

int atn_backward_cuda(at::Tensor top_grad, at::Tensor attn_data,
                               at::Tensor value_data, int kernel_size,
                               int scale_factor,
                               at::Tensor attn_grad, at::Tensor value_grad)
{
  CHECK_INPUT(top_grad);
  CHECK_INPUT(attn_data);
  CHECK_INPUT(value_data);
  CHECK_INPUT(attn_grad);
  CHECK_INPUT(value_grad);
  at::DeviceGuard guard(top_grad.device());

  int batch_size = top_grad.size(0);
  int data_height = top_grad.size(1);
  int data_width = top_grad.size(2);
  int num_channels = top_grad.size(3);

  ATNBackwardLauncher(top_grad, attn_data, value_data, kernel_size,
                             scale_factor, batch_size, num_channels,
                             data_height, data_width, attn_grad, value_grad);

  return 1;
}