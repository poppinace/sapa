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
__global__ void QKForward(const int n,
                          const scalar_t *data_query,
                          const scalar_t *data_key,
                          const int kernel_size,
                          const int scale_factor,
						  const int channels,
                          const int height,
						  const int width,
                          scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, n) {

    int _temp = index;
    const int k = _temp % (kernel_size * kernel_size);
    _temp /= (kernel_size * kernel_size);
    const int qw = _temp % width;
    _temp /= width;
    const int qh = _temp % height;
    _temp /= height;
    const int b = _temp;

    int down_qw = qw / scale_factor;
    int down_qh = qh / scale_factor;
    int key_width = width / scale_factor;
    int key_height = height / scale_factor;
	int kix = k % kernel_size;
	int kiy = k / kernel_size;
	int kh = down_qh - (kernel_size - 1) / 2 + kiy;
	int kw = down_qw - (kernel_size - 1) / 2 + kix;

	const scalar_t* data_query_ptr = data_query + ((b * height + qh) * width + qw) * channels;
    const scalar_t* data_key_ptr = data_key + ((b * key_height + kh) * key_width + kw) * channels;

    scalar_t output_val = 0;
//     if (kh >= 0 && kh <= key_height - 1 && kw >= 0 && kw <= key_width - 1)
// 	    for (int c = 0; c < channels; ++c)
// 		    output_val += data_query_ptr[c] * data_key_ptr[c];
    for (int c = 0; c < channels; ++c){
        if (kh < 0 || kh > key_height - 1 || kw < 0 || kw > key_width - 1){
            continue;
        }
		output_val += data_query_ptr[c] * data_key_ptr[c];
	}
	top_data[index] = output_val;
  }
}


int QKForwardLauncher(const at::Tensor query_data,
						const at::Tensor key_data,
                        const int kernel_size,
                        const int scale_factor,
						const int batch_size,
                        const int channels,
						const int height,
                        const int width,
						at::Tensor output) {
  const int output_size = batch_size * height * width * kernel_size * kernel_size;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      query_data.scalar_type(), "QKLauncherForward", ([&] {
        const scalar_t *data_query = query_data.data_ptr<scalar_t>();
        const scalar_t *data_key = key_data.data_ptr<scalar_t>();
        scalar_t *top_data = output.data_ptr<scalar_t>();

        QKForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, data_query, data_key, kernel_size,
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
__global__ void QKBackward(const int n,
                             const scalar_t *grad_top,
                             const scalar_t *data_query,
                             const scalar_t *data_key,
                             const int kernel_size,
                             const int scale_factor,
                             const int channels,
                             const int height,
                             const int width,
                             scalar_t *grad_query,
                             scalar_t *grad_key) {
	CUDA_1D_KERNEL_LOOP(index, n) {

    int _temp = index;
    const int k = _temp % (kernel_size * kernel_size);
    _temp /= (kernel_size * kernel_size);
    const int qw = _temp % width;
    _temp /= width;
    const int qh = _temp % height;
    _temp /= height;
    const int b = _temp;

    int down_qw = qw / scale_factor;
    int down_qh = qh / scale_factor;
    int key_width = width / scale_factor;
    int key_height = height / scale_factor;
	int kix = k % kernel_size;
	int kiy = k / kernel_size;
	int kh = down_qh - (kernel_size - 1) / 2 + kiy;
	int kw = down_qw - (kernel_size - 1) / 2 + kix;

	int query_offset = ((b * height + qh) * width + qw) * channels;
	int key_offset = ((b * key_height + kh) * key_width + kw) * channels;

	const scalar_t* data_query_ptr = data_query + query_offset;
    const scalar_t* data_key_ptr = data_key + key_offset;

    scalar_t* grad_query_ptr = grad_query + query_offset;
    scalar_t* grad_key_ptr = grad_key + key_offset;

    for (int c = 0; c < channels; ++c) {
        if (kh >= 0 && kh <= key_height - 1 && kw >= 0 && kw <= key_width - 1) {
            atomicAdd(grad_query_ptr++, data_key_ptr[c] * grad_top[index]);
            atomicAdd(grad_key_ptr++, data_query_ptr[c] * grad_top[index]);
        }
	}
  }
}

template <typename scalar_t>
__global__ void QKBackwardSM(const int n,
                             const scalar_t *grad_top,
                             const scalar_t *data_query,
                             const scalar_t *data_key,
                             const int kernel_size,
                             const int scale_factor,
                             const int channels,
                             const int height,
                             const int width,
                             scalar_t *grad_query,
                             scalar_t *grad_key) {
	CUDA_1D_KERNEL_LOOP(index, n) {
	extern __shared__ int _s[];
    scalar_t* cache_grad_query = (scalar_t*)_s;
    unsigned int tid = threadIdx.x;

    int _temp = index;
    const int k = _temp % (kernel_size * kernel_size);
    _temp /= (kernel_size * kernel_size);
    const int qw = _temp % width;
    _temp /= width;
    const int qh = _temp % height;
    _temp /= height;
    const int b = _temp;

    int down_qw = qw / scale_factor;
    int down_qh = qh / scale_factor;
    int key_width = width / scale_factor;
    int key_height = height / scale_factor;
	int kix = k % kernel_size;
	int kiy = k / kernel_size;
	int kh = down_qh - (kernel_size - 1) / 2 + kiy;
	int kw = down_qw - (kernel_size - 1) / 2 + kix;

	int query_offset = ((b * height + qh) * width + qw) * channels;
	int key_offset = ((b * key_height + kh) * key_width + kw) * channels;

	const scalar_t* data_query_ptr = data_query + query_offset;
    const scalar_t* data_key_ptr = data_key + key_offset;

    scalar_t* grad_query_ptr = grad_query + query_offset;
    scalar_t* grad_key_ptr = grad_key + key_offset;

    for (int c = 0; c < channels; ++c) {
        *(cache_grad_query + threadIdx.x)=0;
        if (kh >= 0 && kh <= key_height - 1 && kw >= 0 && kw <= key_width - 1) {
            atomicAdd(grad_key_ptr++, data_query_ptr[c] * grad_top[index]);
            *(cache_grad_query + threadIdx.x) = data_key_ptr[c] * grad_top[index];
        }
        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_q=cache_grad_query[0];
          for (unsigned int tid = 1; tid < blockDim.x; ++tid)
          {
            _grad_q += cache_grad_query[tid];
          }
          grad_query_ptr[c] = _grad_q;
        }
        __syncthreads();
	}
  }
}

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
					   at::Tensor key_grad) {
  const int output_size = batch_size * height * width * kernel_size * kernel_size;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "QKLauncherBackward", ([&] {
        const scalar_t *grad_top = top_grad.data_ptr<scalar_t>();
        const scalar_t *data_query = query_data.data_ptr<scalar_t>();
        const scalar_t *data_key = key_data.data_ptr<scalar_t>();
        scalar_t *grad_query = query_grad.data_ptr<scalar_t>();
        scalar_t *grad_key = key_grad.data_ptr<scalar_t>();

        QKBackwardSM<scalar_t>
            <<<GET_BLOCKS(output_size), kernel_size * kernel_size, kernel_size * kernel_size * sizeof(scalar_t)>>>(
                output_size, grad_top, data_query, data_key, kernel_size,
                scale_factor, channels, height, width, grad_query, grad_key);
      }));

  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}
