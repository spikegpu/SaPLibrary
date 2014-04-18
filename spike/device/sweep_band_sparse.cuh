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
preBck_offDiag_divide(int N, int width, T *pivots, T *dB) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= width)
		return;
	int row_idx = blockIdx.y + blockIdx.z * gridDim.y;
	if (row_idx >= N) return;

	dB[row_idx * width + idx] /= pivots[row_idx];
}

} // namespace sparse
} // namespace device
} // namespace spike

#endif
