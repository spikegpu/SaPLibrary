/** \file factor_band_sparse.cuh
 *  Various forward/backward sweep CUDA kernels used for the case of partitions
 *  for sparse methods.
 */

#ifndef SWEEP_BAND_SPARSE_CUH
#define SWEEP_BAND_SPARSE_CUH

#include <cuda.h>


namespace spike {
namespace device {
namespace sparse {

template <typename T>
__global__ void
fwdElim_spike(int N, int numPartitions, int width, int *row_offsets, int *column_indices, T *values, T *dB, int *left_spike_widths, int *right_spike_widths, int *first_rows) 
{
	int partSize  = N / numPartitions;
	int remainder = N % numPartitions;
	int first_row, last_row, idx, bidy = blockIdx.y;

	if (bidy < gridDim.y / 2) {
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= right_spike_widths[bidy]) return;
		if (bidy < remainder) {
			first_row = bidy * (partSize + 1);
			last_row  = first_row + (partSize + 1);
		} else {
			first_row = bidy * partSize + remainder;
			last_row  = first_row + partSize;
		}
		first_row = first_rows[bidy];
	} else {
		bidy -= gridDim.y / 2 - 1;
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= left_spike_widths[bidy - 1])
			return;

		idx += width - left_spike_widths[bidy-1];
		if (bidy < remainder) {
			first_row = bidy * (partSize + 1);
			last_row  = first_row + (partSize + 1);
		} else {
			first_row = bidy * partSize + remainder;
			last_row  = first_row + partSize;
		}
	}

	int start_idx = row_offsets[first_row + 1];
	int end_idx;
	for (int i = first_row + 1; i < last_row; i++) {
		end_idx = row_offsets[i + 1];
		T tmp_val = dB[i * width + idx];

		for (int l = start_idx; l < end_idx; l++) {
			int cur_k = column_indices[l];
			if (cur_k >= i)
				break;

			tmp_val -= dB[cur_k * width + idx] * values[l];
		}

		dB[i * width + idx] = tmp_val;
		start_idx = end_idx;
	}
}

template <typename T>
__global__ void
fwdElim_spike_partial(int N, int K, int numPartitions, int width, int *lu_row_offsets, int *lu_column_indices, T *lu_values, int *ul_row_offsets, int *ul_column_indices, T *ul_values, T *dB, int *left_spike_widths, int *right_spike_widths) 
{
	int partSize  = N / numPartitions;
	int remainder = N % numPartitions;
	int first_row, last_row, idx, bidy = blockIdx.y;

	if (bidy < gridDim.y / 2) {
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= right_spike_widths[bidy]) return;
		if (bidy < remainder) {
			first_row = bidy * (partSize + 1);
			last_row  = first_row + (partSize + 1);
		} else {
			first_row = bidy * partSize + remainder;
			last_row  = first_row + partSize;
		}
		first_row = last_row - K;

		int start_idx = lu_row_offsets[first_row + 1];
		int end_idx;
		for (int i = first_row + 1; i < last_row; i++) {
			end_idx = lu_row_offsets[i + 1];
			T tmp_val = dB[(i-first_row + bidy * K) * width + idx];

			for (int l = start_idx; l < end_idx; l++) {
				int cur_k = lu_column_indices[l];
				if (cur_k < first_row)
					continue;

				if (cur_k >= i)
					break;

				tmp_val -= dB[(cur_k-first_row + bidy * K) * width + idx] * lu_values[l];
			}

			dB[(i - first_row + bidy * K) * width + idx] = tmp_val;
			start_idx = end_idx;
		}
	} else {
		bidy -= gridDim.y / 2 - 1;
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= left_spike_widths[bidy - 1])
			return;

		idx += width - left_spike_widths[bidy-1];
		if (bidy < remainder)
			first_row = bidy * (partSize + 1);
		else
			first_row = bidy * partSize + remainder;
		last_row = first_row + K;

		int start_idx;
		int end_idx = ul_row_offsets[last_row - 1];
		for (int i = last_row - 2; i >= first_row; i--) {
			start_idx = ul_row_offsets[i];
			T tmp_val = dB[(i-first_row + bidy * K) * width + idx];

			for (int l = end_idx - 1; l >= start_idx; l--) {
				int cur_k = ul_column_indices[l];
				if (cur_k >= last_row)
					continue;

				if (cur_k <= i)
					break;

				tmp_val -= dB[(cur_k-first_row + bidy * K) * width + idx] * ul_values[l];
			}

			dB[(i - first_row + bidy * K) * width + idx] = tmp_val;
			end_idx = start_idx;
		}
	}

}

template <typename T>
__global__ void
bckElim_spike(int N, int numPartitions, int width, int *row_offsets, int *column_indices, T *values, T *dB, int *left_spike_widths, int *right_spike_widths, int *first_rows) 
{
	int partSize  = N / numPartitions;
	int remainder = N % numPartitions;
	int first_row, last_row, idx, bidy = blockIdx.y;

	if (bidy < gridDim.y / 2) {
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= right_spike_widths[bidy]) return;
		if (bidy < remainder) {
			first_row = bidy * (partSize + 1);
			last_row  = first_row + (partSize + 1);
		} else {
			first_row = bidy * partSize + remainder;
			last_row  = first_row + partSize;
		}
		first_row = first_rows[bidy];
	} else {
		bidy -= gridDim.y / 2 - 1;
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= left_spike_widths[bidy - 1])
			return;

		idx += width - left_spike_widths[bidy-1];
		if (bidy < remainder) {
			first_row = bidy * (partSize + 1);
			last_row  = first_row + (partSize + 1);
		} else {
			first_row = bidy * partSize + remainder;
			last_row  = first_row + partSize;
		}
	}

	int start_idx;
	int end_idx = row_offsets[last_row - 1];
	for (int i = last_row - 2; i >= first_row; i--) {
		start_idx = row_offsets[i];
		T tmp_val = dB[i * width + idx];

		for (int l = end_idx - 1; l >= start_idx; l--) {
			int cur_k = column_indices[l];
			if (cur_k <= i)
				break;

			tmp_val -= dB[cur_k * width + idx] * values[l];
		}

		dB[i * width + idx] = tmp_val;
		end_idx = start_idx;
	}
}

template <typename T>
__global__ void
bckElim_spike_partial(int N, int K, int numPartitions, int width, int *lu_row_offsets, int *lu_column_indices, T *lu_values, int *ul_row_offsets, int *ul_column_indices, T *ul_values, T *dB, int *left_spike_widths, int *right_spike_widths)
{
	int partSize  = N / numPartitions;
	int remainder = N % numPartitions;
	int first_row, last_row, idx, bidy = blockIdx.y;

	if (bidy < gridDim.y / 2) {
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= right_spike_widths[bidy]) return;
		if (bidy < remainder) {
			first_row = bidy * (partSize + 1);
			last_row  = first_row + (partSize + 1);
		} else {
			first_row = bidy * partSize + remainder;
			last_row  = first_row + partSize;
		}
		first_row = last_row - K;

		int start_idx;
		int end_idx = lu_row_offsets[last_row - 1];
		for (int i = last_row - 2; i >= first_row; i--) {
			start_idx = lu_row_offsets[i];
			T tmp_val = dB[(i - first_row + bidy * K) * width + idx];

			for (int l = end_idx - 1; l >= start_idx; l--) {
				int cur_k = lu_column_indices[l];
				if (cur_k >= last_row)
					continue;

				if (cur_k <= i)
					break;

				tmp_val -= dB[(cur_k - first_row + bidy * K)* width + idx] * lu_values[l];
			}

			dB[(i - first_row + bidy * K)* width + idx] = tmp_val;
			end_idx = start_idx;
		}
	} else {
		bidy -= gridDim.y / 2 - 1;
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= left_spike_widths[bidy - 1])
			return;

		idx += width - left_spike_widths[bidy-1];
		if (bidy < remainder)
			first_row = bidy * (partSize + 1);
		else 
			first_row = bidy * partSize + remainder;

		last_row = first_row + K;

		int start_idx = ul_row_offsets[first_row + 1];
		int end_idx;
		for (int i = first_row + 1; i < last_row; i++) {
			end_idx = ul_row_offsets[i + 1];
			T tmp_val = dB[(i - first_row + bidy * K) * width + idx];

			for (int l = start_idx; l < end_idx; l++) {
				int cur_k = ul_column_indices[l];
				if (cur_k < first_row)
					continue;

				if (cur_k >= i)
					break;

				tmp_val -= dB[(cur_k - first_row + bidy * K)* width + idx] * ul_values[l];
			}

			dB[(i - first_row + bidy * K)* width + idx] = tmp_val;
			start_idx = end_idx;
		}
	}

}

template <typename T>
__global__ void
preBck_offDiag_divide(int N, int width, T *pivots, T *dB) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= width)
		return;
	int row_idx = blockIdx.y + blockIdx.z * gridDim.y;
	if (row_idx >= N) return;

	dB[row_idx * width + idx] /= pivots[row_idx];
}

template <typename T>
__global__ void
preBck_offDiag_divide_partial(int N, int K, int numPartitions, int width, int rightWidth, T *pivots, T *pivots_ul, T *dB) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= width)
		return;
	int row_idx = blockIdx.y + blockIdx.z * gridDim.y;
	if (row_idx >= K * numPartitions) return;

	int partId    = row_idx / K;
	int partSize  = N / numPartitions;
	int remainder = N % numPartitions;

	int first_row, last_row;

	if (partId < remainder) {
		first_row = partId * (partSize + 1);
		last_row  = first_row + (partSize + 1);
	} else {
		first_row = partId * partSize + remainder;
		last_row  = first_row + partSize;
	}

	if (idx < rightWidth) {
		first_row = last_row - K;
		dB[row_idx * width + idx] /= pivots[first_row + row_idx - partId * K];
	} else {
		dB[row_idx * width + idx] /= pivots_ul[first_row + row_idx - partId * K];
	}


}

} // namespace sparse
} // namespace device
} // namespace spike

#endif
