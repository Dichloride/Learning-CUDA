#include <vector>
#include <cmath>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../tester/utils.h"

template <typename T>
__global__ void traceKernel(const T* input, T* out, size_t rows, size_t cols, size_t diag);

template <typename T>
__global__ void flashAttentionKernel(const T* q, const T* k, const T* v, T* o,
                                     int batch_size, int target_seq_len, int src_seq_len,
                                     int query_heads, int kv_heads, int head_dim, bool is_causal);

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  if (rows == 0 || cols == 0) {
    return T(0);
  }

  const size_t diag = rows < cols ? rows : cols;
  const size_t count = rows * cols;

  T* d_input = nullptr;
  T* d_out = nullptr;

  RUNTIME_CHECK(cudaMalloc(&d_input, count * sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), count * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMalloc(&d_out, sizeof(T)));
  RUNTIME_CHECK(cudaMemset(d_out, 0, sizeof(T)));

  const int threads = 256;
  const int blocks = static_cast<int>((diag + threads - 1) / threads);

  traceKernel<<<blocks, threads>>>(d_input, d_out, rows, cols, diag);
  RUNTIME_CHECK(cudaGetLastError());

  T h_out = T(0);
  RUNTIME_CHECK(cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_out));

  return h_out;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  const size_t q_size = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t kv_size = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  const size_t o_size = q_size;

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_o = nullptr;

  RUNTIME_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, kv_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, kv_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));

  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));

  const int total = batch_size * target_seq_len * query_heads;
  const int threads = 1;
  const int blocks = total;

  const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);
  flashAttentionKernel<<<blocks, threads, shared_bytes>>>(
      d_q, d_k, d_v, d_o,
      batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());

  if (h_o.size() != o_size) {
    h_o.resize(o_size);
  }
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

template <typename T>
__device__ __forceinline__ float toFloat(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float toFloat<half>(half v) {
  return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v) {
  return static_cast<T>(v);
}

template <>
__device__ __forceinline__ half fromFloat<half>(float v) {
  return __float2half_rn(v);
}

template <typename T>
__global__ void traceKernel(const T* input, T* out, size_t rows, size_t cols, size_t diag) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= diag) {
    return;
  }
  const size_t offset = idx * cols + idx;
  if constexpr (std::is_same<T, int>::value) {
    atomicAdd(reinterpret_cast<int*>(out), static_cast<int>(input[offset]));
  } else {
    atomicAdd(reinterpret_cast<float*>(out), static_cast<float>(input[offset]));
  }
}

template <typename T>
__global__ void flashAttentionKernel(const T* q, const T* k, const T* v, T* o,
                                     int batch_size, int target_seq_len, int src_seq_len,
                                     int query_heads, int kv_heads, int head_dim, bool is_causal) {
  extern __shared__ float acc[];
  const int idx = blockIdx.x;
  const int total = batch_size * target_seq_len * query_heads;
  if (idx >= total) {
    return;
  }

  const int qh = idx % query_heads;
  const int tmp = idx / query_heads;
  const int t = tmp % target_seq_len;
  const int b = tmp / target_seq_len;

  const int q_per_kv = query_heads / kv_heads;
  const int kv_head = q_per_kv > 0 ? (qh / q_per_kv) : 0;

  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  float max_score = -CUDART_INF_F;
  const size_t q_base = ((static_cast<size_t>(b) * target_seq_len + t) * query_heads + qh) * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    acc[d] = 0.0f;
  }
  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) {
      continue;
    }
    float dot = 0.0f;
    const size_t k_base = ((static_cast<size_t>(b) * src_seq_len + s) * kv_heads + kv_head) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      dot += toFloat(q[q_base + d]) * toFloat(k[k_base + d]);
    }
    const float score = dot * scale;
    if (score > max_score) {
      max_score = score;
    }
  }

  float sum_exp = 0.0f;
  for (int s = 0; s < src_seq_len; ++s) {
    if (is_causal && s > t) {
      continue;
    }
    float dot = 0.0f;
    const size_t k_base = ((static_cast<size_t>(b) * src_seq_len + s) * kv_heads + kv_head) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      dot += toFloat(q[q_base + d]) * toFloat(k[k_base + d]);
    }
    const float score = dot * scale;
    const float exp_score = expf(score - max_score);
    sum_exp += exp_score;

    const size_t v_base = ((static_cast<size_t>(b) * src_seq_len + s) * kv_heads + kv_head) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      acc[d] += exp_score * toFloat(v[v_base + d]);
    }
  }

  if (sum_exp > 0.0f) {
    const float inv_sum = 1.0f / sum_exp;
    const size_t o_base = q_base;
    for (int d = 0; d < head_dim; ++d) {
      o[o_base + d] = fromFloat<T>(acc[d] * inv_sum);
    }
  } else {
    const size_t o_base = q_base;
    for (int d = 0; d < head_dim; ++d) {
      o[o_base + d] = fromFloat<T>(0.0f);
    }
  }
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
