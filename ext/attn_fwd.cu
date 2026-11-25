// Kernel 1 forward: fused block-causal attention with QK RMS-norm, FP16 tensor cores (WMMA), FP32 softmax. sm_70.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
constexpr int M = 128, N = 64, D = 64, WARPS = 8;
struct Str { long b, l, h; };   // element strides of a [B,L,H,64] tensor (last dim contiguous)

// ---- pre-pass: x -> rms_norm(x) * s in FP32, stored FP16 as [B,H,Lp,64]; rows >= L are zero ----
__global__ void norm_k(const half* __restrict__ x, Str st, half* __restrict__ out, const float* s, int B, int H, int L, int Lp) {
  const int lane = threadIdx.x & 31, tok = blockIdx.x * WARPS + (threadIdx.x >> 5);
  const int b = tok / (H * Lp), rem = tok - b * H * Lp, h = rem / Lp, l = rem - h * Lp;
  if (b >= B) return;
  half2* dst = (half2*)(out + (long)tok * D) + lane;
  if (l >= L) { *dst = __float2half2_rn(0.f); return; }
  const half2 v = ((const half2*)(x + b * st.b + l * st.l + h * st.h))[lane];
  float a0 = __low2float(v), a1 = __high2float(v), ss = a0 * a0 + a1 * a1;
  for (int k = 16; k; k >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, k);
  const float r = 1.f / sqrtf(ss * (1.f / D) + 1.1920928955078125e-7f);
  a0 *= r; a1 *= r;
  if (s) { a0 *= s[lane * 2]; a1 *= s[lane * 2 + 1]; }
  *dst = __floats2half2_rn(a0, a1);
}

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> FA;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> FBc;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> FBr;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> FC;

// one value per accumulator element -> per-row reduction across the warp, result back per element
template <bool MAX> __device__ __forceinline__ void row_reduce(float* v, const int* id, float* buf, float* rowv, int lane) {
  for (int e = 0; e < 8; e++) buf[id[e]] = v[e];
  __syncwarp();
  if (lane < 16) {
    float a = buf[lane * 16];
    for (int c = 1; c < 16; c++) { float x = buf[lane * 16 + c]; a = MAX ? fmaxf(a, x) : a + x; }
    rowv[lane] = a;
  }
  __syncwarp();
  for (int e = 0; e < 8; e++) v[e] = rowv[id[e] >> 4];
}

template <bool SC> __global__ void attn_k(const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v, Str sv,
                                          half* __restrict__ o, const int* lens, int H, int L, int Lp, int P) {
  __shared__ half Ks[N * D], Vs[N * D], Ps[WARPS * 16 * 16];
  __shared__ float buf[WARPS * 256], rowv[WARPS * 16], idx[256];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int nt = Lp / M, b = blockIdx.x / (H * nt), rem = blockIdx.x - b * H * nt, h = rem / nt, row0 = (rem - h * nt) * M;
  const int len = lens ? min(lens[b], L) : L;
  const int rmax = min(row0 + M, len) - 1;
  half* op = o + ((long)b * L * H + h) * D;                        // output [B,L,H,64] contiguous
  idx[tid] = (float)tid;
  __syncthreads();
  FC idf; wmma::load_matrix_sync(idf, idx, 16, wmma::mem_row_major);
  int id[8]; for (int e = 0; e < 8; e++) id[e] = (int)idf.x[e];
  const int r0 = row0 + 16 * w;
  FC O[4]; for (int d = 0; d < 4; d++) wmma::fill_fragment(O[d], 0.f);
  float sum[8], mx[8]; int lim[8];
  for (int e = 0; e < 8; e++) {
    int ri = r0 + (id[e] >> 4);
    lim[e] = ri < len ? min((ri / P + 1) * P, len) : 0;
    sum[e] = 0.f; mx[e] = -INFINITY;
  }
  if (rmax >= row0) {
    const int lmax = min((rmax / P + 1) * P, len);
    const int lmin = row0 + M <= len ? min((row0 / P + 1) * P, len) : 0;
    const long base = ((long)b * H + h) * Lp * D;
    FA Qf[4]; for (int d = 0; d < 4; d++) wmma::load_matrix_sync(Qf[d], q + base + (long)r0 * D + d * 16, D);
    const uint4* kg = (const uint4*)(k + base);
    const half* vg = v + b * sv.b + h * sv.h;
    float* mybuf = buf + w * 256; float* myrow = rowv + w * 16; half* myP = Ps + w * 256;
    for (int j0 = 0; j0 < lmax; j0 += N) {
      __syncthreads();
      for (int i = tid; i < N * D / 8; i += M * 2) {               // 512 uint4 per tile, 256 threads
        int row = i >> 3, c = i & 7;
        ((uint4*)Ks)[i] = kg[(long)(j0 + row) * 8 + c];
        ((uint4*)Vs)[i] = j0 + row < L ? ((const uint4*)(vg + (long)(j0 + row) * sv.l))[c] : make_uint4(0, 0, 0, 0);
      }
      __syncthreads();
      const bool nm = j0 + N > lmin;
      for (int n = 0; n < 4; n++) {                                 // 16 keys at a time: S is one fragment, not four
        FC S; wmma::fill_fragment(S, 0.f);
        for (int d = 0; d < 4; d++) { FBc kb; wmma::load_matrix_sync(kb, Ks + n * 16 * D + d * 16, D); 
          wmma::mma_sync(S, Qf[d], kb, S); }
        const int jb = j0 + n * 16;
        if (SC) {
          float m[8];
          for (int e = 0; e < 8; e++) m[e] = !nm || jb + (id[e] & 15) < lim[e] ? S.x[e] * 0.125f : -INFINITY;
          row_reduce<true>(m, id, mybuf, myrow, lane);
          for (int e = 0; e < 8; e++) {
            float nw = fmaxf(mx[e], m[e]), f = mx[e] == nw ? 1.f : expf(mx[e] - nw);
            sum[e] *= f; for (int d = 0; d < 4; d++) O[d].x[e] *= f;
            mx[e] = nw;
          }
        }
        for (int e = 0; e < 8; e++) {
          float x = S.x[e] * 0.125f;
          if (SC) x -= mx[e];
          half ph = __float2half(expf(x));
          if (nm && jb + (id[e] & 15) >= lim[e]) ph = __float2half(0.f);
          sum[e] += __half2float(ph);
          myP[(id[e] >> 4) * 16 + (id[e] & 15)] = ph;
        }
        __syncwarp();
        FA pf; wmma::load_matrix_sync(pf, myP, 16);
        __syncwarp();
        for (int d = 0; d < 4; d++) { FBr vb; wmma::load_matrix_sync(vb, Vs + n * 16 * D + d * 16, D); wmma::mma_sync(O[d], pf, vb, O[d]); }
      }
    }
    row_reduce<false>(sum, id, mybuf, myrow, lane);
  }
  for (int e = 0; e < 8; e++) {
    int ri = r0 + (id[e] >> 4);
    if (ri >= L) continue;
    const float inv = 1.f / sum[e];
    for (int d = 0; d < 4; d++) op[(long)ri * H * D + d * 16 + (id[e] & 15)] = __float2half(lim[e] ? O[d].x[e] * inv : 0.f);
  }
}

static Str strides(const at::Tensor& t) { return {t.stride(0), t.stride(1), t.stride(2)}; }
static at::Tensor prep(const at::Tensor& t) {
  bool ok = t.stride(3) == 1 && ((uintptr_t)t.data_ptr() % 16 == 0) && t.stride(0) % 8 == 0 && t.stride(1) % 8 == 0 && t.stride(2) % 8 == 0;
  return ok ? t : t.contiguous();
}
static void chk() { cudaError_t e = cudaGetLastError(); TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e)); }

at::Tensor attn_fwd(at::Tensor q, at::Tensor k, at::Tensor v, long P, c10::optional<at::Tensor> lens, c10::optional<at::Tensor> sq, c10::optional<at::Tensor> sk) {
  TORCH_CHECK(q.dim() == 4 && q.size(3) == D && q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q,k,v must be [B,L,H,64] and equal");
  for (auto& t : {q, k, v}) TORCH_CHECK(t.is_cuda() && t.dtype() == at::kHalf, "q,k,v must be cuda FP16");
  const int B = q.size(0), L = q.size(1), H = q.size(2), Lp = (L + M - 1) / M * M;
  TORCH_CHECK(P >= 1 && B > 0 && L > 0 && H > 0); const int Pi = (int)std::min<long>(P, L);
  TORCH_CHECK((long)B * H * Lp < (1L << 31) - 8);
  at::Tensor ln, s1, s2;
  if (lens) { TORCH_CHECK(lens->numel() == B); ln = lens->to(q.device(), at::kInt).contiguous(); }
  if (sq) { TORCH_CHECK(sq->numel() == D); s1 = sq->to(q.device(), at::kFloat).contiguous(); }
  if (sk) { TORCH_CHECK(sk->numel() == D); s2 = sk->to(q.device(), at::kFloat).contiguous(); }
  q = prep(q); k = prep(k); v = prep(v);
  auto st = c10::cuda::getCurrentCUDAStream().stream();
  at::Tensor qs = at::empty({B, H, Lp, D}, q.options()), ks = at::empty({B, H, Lp, D}, q.options()), o = at::empty({B, L, H, D}, q.options());
  const int nb = (int)(((long)B * H * Lp + WARPS - 1) / WARPS);
  norm_k<<<nb, 256, 0, st>>>(q.data_ptr<c10::Half>() ? (const half*)q.data_ptr() : nullptr, strides(q), (half*)qs.data_ptr(), s1.defined() ? s1.data_ptr<float>() : nullptr, B, H, L, Lp);
  norm_k<<<nb, 256, 0, st>>>((const half*)k.data_ptr(), strides(k), (half*)ks.data_ptr(), s2.defined() ? s2.data_ptr<float>() : nullptr, B, H, L, Lp);
  const int nblk = B * H * (Lp / M); const int* lp = ln.defined() ? ln.data_ptr<int>() : nullptr;
  if (sq || sk) attn_k<true><<<nblk, 256, 0, st>>>((const half*)qs.data_ptr(), (const half*)ks.data_ptr(), (const half*)v.data_ptr(), strides(v), (half*)o.data_ptr(), lp, H, L, Lp, Pi);
  else attn_k<false><<<nblk, 256, 0, st>>>((const half*)qs.data_ptr(), (const half*)ks.data_ptr(), (const half*)v.data_ptr(), strides(v), (half*)o.data_ptr(), lp, H, L, Lp, Pi);
  chk(); return o;
}
