/*
   CUDA extension for SAPA
   Modified from CARAFE https://github.com/myownskyW7/CARAFE
*/
#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

int QKForwardLauncher(const at::Tensor query_data,
                        const at::Tensor key_data,
                        const int kernel_size,
                        const int scale_factor,
                        const int batch_size,
                        const int channels,
                        const int height,
                        const int width,
                        at::Tensor output);

int QKBackwardLauncher(const at::Tensor top_grad,
                        const at::Tensor query_data,
                        const at::Tensor key_data,
                        const int kernel_size,
                        const int scale_factor,
                        const int batch_size,
                        const int channels,
                        const int height,
                        const int width,
                        at::Tensor query_grad,
                        at::Tensor key_grad);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int qk_forward_cuda(at::Tensor query_data, at::Tensor key_data,
                              int kernel_size, int scale_factor,
                              at::Tensor output)
{
  CHECK_INPUT(query_data);
  CHECK_INPUT(key_data);
  CHECK_INPUT(output);
  at::DeviceGuard guard(query_data.device());

  int batch_size = query_data.size(0);
  int data_height = query_data.size(1);
  int data_width = query_data.size(2);
  int num_channels = query_data.size(3);

  QKForwardLauncher(query_data, key_data, kernel_size,
                            scale_factor, batch_size, num_channels, data_height,
                            data_width, output);

  return 1;
}

int qk_backward_cuda(at::Tensor top_grad, at::Tensor query_data,
                               at::Tensor key_data, int kernel_size,
                               int scale_factor,
                               at::Tensor query_grad, at::Tensor key_grad)
{
  CHECK_INPUT(top_grad);
  CHECK_INPUT(query_data);
  CHECK_INPUT(key_data);
  CHECK_INPUT(query_grad);
  CHECK_INPUT(key_grad);
  at::DeviceGuard guard(top_grad.device());

  int batch_size = query_data.size(0);
  int data_height = query_data.size(1);
  int data_width = query_data.size(2);
  int num_channels = query_data.size(3);

  QKBackwardLauncher(top_grad, query_data, key_data, kernel_size,
                             scale_factor, batch_size, num_channels,
                             data_height, data_width, query_grad, key_grad);

  return 1;
}
