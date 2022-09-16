/*
   CUDA extension for SAPA
   Modified from CARAFE https://github.com/myownskyW7/CARAFE
*/
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

using namespace at;  // temporal fix for pytorch<=0.4.1 (see #9848)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65536;
  return min(optimal_block_num, max_block_num);
}
template <typename scalar_t>
__global__ void ATNForward(const int n,
                          const scalar_t *data_attn,
                          const scalar_t *data_value,
                          const int kernel_size,
                          const int scale_factor,
                          const int channels,
                          const int height,
                          const int width,
                          scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c = _temp % channels;
    _temp /= channels;
    const int qw = _temp % width;
    _temp /= width;
    const int qh = _temp % height;
    _temp /= height;
    const int b = _temp;

    int down_qw = qw / scale_factor;
    int down_qh = qh / scale_factor;
    int key_width = width / scale_factor;
    int key_height = height / scale_factor;
    int start_w = down_qw - (kernel_size - 1) / 2;
    int end_w = down_qw + (kernel_size - 1) / 2 + 1;
    int start_h = down_qh - (kernel_size - 1) / 2;
    int end_h = down_qh + (kernel_size - 1) / 2 + 1;

    scalar_t output_val = 0;
    for (int iy = start_h; iy < end_h; iy++) {
      for (int ix = start_w; ix < end_w; ix++) {

        const int kiy = iy - down_qh + (kernel_size - 1) / 2;
        const int kix = ix - down_qw + (kernel_size - 1) / 2;
        const int ac = kiy * kernel_size + kix;
        const scalar_t* data_attn_ptr = data_attn + ((b * height + qh) * width + qw) * kernel_size * kernel_size + ac;
        const scalar_t* data_value_ptr = data_value + ((b * key_height + iy) * key_width + ix) * channels + c;

        if (iy >= 0 && iy <= key_height - 1 && ix >= 0 && ix <= key_width - 1) {
          output_val += *data_attn_ptr * *data_value_ptr;
        }
      }
    }
    top_data[index] = output_val;
  }
}

int ATNForwardLauncher(const at::Tensor attn_data,
                       const at::Tensor value_data,
                       const int kernel_size,
                       const int scale_factor,
                       const int batch_size,
                       const int channels,
                       const int height,
                       const int width,
                       at::Tensor output) {
  const int output_size = batch_size * height * width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      attn_data.scalar_type(), "ATNLauncherForward", ([&] {
        const scalar_t *data_attn = attn_data.data_ptr<scalar_t>();
        const scalar_t *data_value = value_data.data_ptr<scalar_t>();
        scalar_t *top_data = output.data_ptr<scalar_t>();

        ATNForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, data_attn, data_value, kernel_size,
                scale_factor, channels, height, width, top_data);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}

template <typename scalar_t>
__global__ void ATNBackward(const int n,
                            const scalar_t *grad_top,
                            const scalar_t *data_attn,
                            const scalar_t *data_value,
                            const int kernel_size,
                            const int scale_factor,
                            const int channels,
                            const int height,
                            const int width,
                            scalar_t *grad_attn,
                            scalar_t *grad_value) {
  CUDA_1D_KERNEL_LOOP(index, n) {

    int _temp = index;
    const int c = _temp % channels;
    _temp /= channels;
    const int qw = _temp % width;
    _temp /= width;
    const int qh = _temp % height;
    _temp /= height;
    const int b = _temp;

    int down_qw = qw / scale_factor;
    int down_qh = qh / scale_factor;
    int key_width = width / scale_factor;
    int key_height = height / scale_factor;
    int start_w = down_qw - (kernel_size - 1) / 2;
    int end_w = down_qw + (kernel_size - 1) / 2 + 1;
    int start_h = down_qh - (kernel_size - 1) / 2;
    int end_h = down_qh + (kernel_size - 1) / 2 + 1;

    for (int iy = start_h; iy < end_h; iy++) {
      for (int ix = start_w; ix < end_w; ix++) {

        const int kiy = iy - down_qh + (kernel_size - 1) / 2;
        const int kix = ix - down_qw + (kernel_size - 1) / 2;
        const int ac = kiy * kernel_size + kix;
        const int attn_offset = ((b * height + qh) * width + qw) * kernel_size * kernel_size + ac;
        const int value_offset = ((b * key_height + iy) * key_width + ix) * channels + c;
        const scalar_t* data_attn_ptr = data_attn + attn_offset;
        const scalar_t* data_value_ptr = data_value + value_offset;

        scalar_t* grad_attn_ptr = grad_attn + attn_offset;
        scalar_t* grad_value_ptr = grad_value + value_offset;

        if (iy >= 0 && iy <= key_height - 1 && ix >= 0 && ix <= key_width - 1) {
            atomicAdd(grad_attn_ptr, *data_value_ptr * grad_top[index]);
            atomicAdd(grad_value_ptr, *data_attn_ptr * grad_top[index]);
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void ATNBackwardSM(const int n,
                            const scalar_t *grad_top,
                            const scalar_t *data_attn,
                            const scalar_t *data_value,
                            const int kernel_size,
                            const int scale_factor,
                            const int channels,
                            const int height,
                            const int width,
                            scalar_t *grad_attn,
                            scalar_t *grad_value) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    extern __shared__ int _s[];
    scalar_t* cache_grad_attn = (scalar_t*)_s;
    unsigned int tid = threadIdx.x;

    int _temp = index;
    const int c = _temp % channels;
    _temp /= channels;
    const int qw = _temp % width;
    _temp /= width;
    const int qh = _temp % height;
    _temp /= height;
    const int b = _temp;

    int down_qw = qw / scale_factor;
    int down_qh = qh / scale_factor;
    int key_width = width / scale_factor;
    int key_height = height / scale_factor;
    int start_w = down_qw - (kernel_size - 1) / 2;
    int end_w = down_qw + (kernel_size - 1) / 2 + 1;
    int start_h = down_qh - (kernel_size - 1) / 2;
    int end_h = down_qh + (kernel_size - 1) / 2 + 1;

    for (int iy = start_h; iy < end_h; iy++) {
      for (int ix = start_w; ix < end_w; ix++) {

        const int kiy = iy - down_qh + (kernel_size - 1) / 2;
        const int kix = ix - down_qw + (kernel_size - 1) / 2;
        const int ac = kiy * kernel_size + kix;
        const int attn_offset = ((b * height + qh) * width + qw) * kernel_size * kernel_size + ac;
        const int value_offset = ((b * key_height + iy) * key_width + ix) * channels + c;
        const scalar_t* data_attn_ptr = data_attn + attn_offset;
        const scalar_t* data_value_ptr = data_value + value_offset;

        scalar_t* grad_attn_ptr = grad_attn + attn_offset;
        scalar_t* grad_value_ptr = grad_value + value_offset;

        *(cache_grad_attn + threadIdx.x) = 0;

        if (iy >= 0 && iy <= key_height - 1 && ix >= 0 && ix <= key_width - 1) {
            atomicAdd(grad_value_ptr, *data_attn_ptr * grad_top[index]);
            *(cache_grad_attn + threadIdx.x) = *data_value_ptr * grad_top[index];
        }
        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1)
        {
          if (tid < s) {
            cache_grad_attn[tid] += cache_grad_attn[tid + s];
            if (tid + (s << 1) < spre)
            {
              cache_grad_attn[tid] += cache_grad_attn[tid + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          *grad_attn_ptr = cache_grad_attn[0];
        }
        __syncthreads();
      }
    }
  }
}

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
                        at::Tensor value_grad) {
  const int output_size = batch_size * height * width * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "ATNLauncherBackward", ([&] {
        const scalar_t *grad_top = top_grad.data_ptr<scalar_t>();
        const scalar_t *data_attn = attn_data.data_ptr<scalar_t>();
        const scalar_t *data_value = value_data.data_ptr<scalar_t>();
        scalar_t *grad_attn = attn_grad.data_ptr<scalar_t>();
        scalar_t *grad_value = value_grad.data_ptr<scalar_t>();

        ATNBackwardSM<scalar_t>
            <<<GET_BLOCKS(output_size), channels, channels * sizeof(scalar_t)>>>(
                output_size, grad_top, data_attn, data_value, kernel_size,
                scale_factor, channels, height, width, grad_attn, grad_value);
      }));

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}