// STREAM triad a = b + s*c on float buffers; toolchain proof and HBM bandwidth ceiling.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void triad_k(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, float s, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = b[i] + s * c[i];  // one element per thread, contiguous blocks (BabelStream style)
}
void triad(at::Tensor a, at::Tensor b, at::Tensor c, double s) {
  TORCH_CHECK(a.is_cuda() && a.is_contiguous() && b.is_contiguous() && c.is_contiguous() && a.dtype() == at::kFloat);
  TORCH_CHECK(a.numel() == b.numel() && a.numel() == c.numel());
  long n = a.numel();
  int tpb = 1024, blocks = (int)((n + tpb - 1) / tpb);
  triad_k<<<blocks, tpb, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), (float)s, n);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}
at::Tensor attn_fwd(at::Tensor, at::Tensor, at::Tensor, long, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("triad", &triad); m.def("attn_fwd", &attn_fwd); }
