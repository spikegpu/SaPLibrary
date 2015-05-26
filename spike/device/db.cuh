/** \file data_transfer.cuh
 *
 * CUDA kernels for DB
 */

#ifndef DB_CUH
#define DB_CUH

namespace spike {
namespace device {

template<typename T>
__global__ void
getResidualValues(int        N,
		          T*         c_vals,
				  T*         max_vals,
				  const int* row_ptr)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x;

	if (bid >= N)
		return;

	__shared__ int start;
	__shared__ int end;
	__shared__ T max_val;

	if (threadIdx.x == 0) {
		start = row_ptr[bid];
		end = row_ptr[bid + 1];
		max_val = max_vals[bid];
	}
	__syncthreads();

	for (int i = threadIdx.x + start; i < end; i += blockDim.x)
		c_vals[i] = log(max_val / c_vals[i]);
}

template <typename T>
__device__
T getMinValueAndIndex(const volatile T &v1, const volatile T & v2, const volatile int &i1, const volatile int &i2,volatile int &idx) 
{
	if (v1 < v2) {
		idx = i1;
		return v1;
	}
	idx = i2;
	return v2;
}

template <typename T>
__global__ void
findInitialMatch(int N, int *row_offsets, int *column_indices, T *values, T *u_values, T *v_values,
		         int *matches, int *rev_matches, bool *matched, bool *rev_matched)
{
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int tid = threadIdx.x;
	if (bid >= N)
		return;

	__shared__ volatile T   min_values[64];
	__shared__ volatile int min_indices[64];

	min_values[tid] = 1e37;

	int start_idx = row_offsets[bid], end_idx = row_offsets[bid + 1];

	for (int l = start_idx + tid; l < end_idx; l += blockDim.x) {
		int column = column_indices[l];
		T   tmp_val = values[l] - u_values[column];
		if (min_values[tid] > tmp_val) {
			min_values[tid]  = tmp_val;
			min_indices[tid] = l;
		}
	}

	if (threadIdx.x >= 32)
		return;

	__syncthreads();

	min_values[tid] = getMinValueAndIndex(min_values[tid], min_values[tid+32], min_indices[tid], min_indices[tid + 32], min_indices[tid]);
	min_values[tid] = getMinValueAndIndex(min_values[tid], min_values[tid+16], min_indices[tid], min_indices[tid + 16], min_indices[tid]);
	min_values[tid] = getMinValueAndIndex(min_values[tid], min_values[tid+8], min_indices[tid], min_indices[tid + 8], min_indices[tid]);
	min_values[tid] = getMinValueAndIndex(min_values[tid], min_values[tid+4], min_indices[tid], min_indices[tid + 4], min_indices[tid]);
	min_values[tid] = getMinValueAndIndex(min_values[tid], min_values[tid+2], min_indices[tid], min_indices[tid + 2], min_indices[tid]);

	if (tid == 0) {
		v_values[bid] = (min_values[0] = getMinValueAndIndex(min_values[0], min_values[1], min_indices[0], min_indices[1], min_indices[0]));
		int column = column_indices[min_indices[0]];
		if (!matched[column]) {
			rev_matched[bid] = true;
			matched[column]  = true;
			matches[column]  = bid;
			rev_matches[bid] = min_indices[0];
		}
	}
}

} // namespace device
} // namespace spike


#endif
