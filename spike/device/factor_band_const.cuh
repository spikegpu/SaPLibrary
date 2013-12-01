/** \file factor_band_var.cuh
 *  Various matrix factorization CUDA kernels used for the case of partitions
    with equal bandwidths.
 */

#ifndef FACTOR_BAND_CONST_H
#define FACTOR_BAND_CONST_H

#include <cuda.h>
#include <spike/common.h>


namespace spike {
namespace device {

template <typename T>
__device__ inline T
boostValue(const T in_val, T &out_val, const T threshold) {
	if (in_val > threshold || in_val < -threshold)
		return in_val;
	if (in_val < 0) {
		out_val = -threshold;
		return -threshold;
	}
	out_val = threshold;
	return threshold;
}

template <typename T>
__device__ inline T
boostValue(const T in_val, T &out_val, const T threshold, const T new_val) {
	if (in_val > threshold || in_val < -threshold)
		return in_val;
	if (in_val < 0) {
		out_val = -new_val;
		return -new_val;
	}
	out_val = new_val;
	return new_val;
}

template <typename T>
__global__ void
bandLU(T *dA, int k, int partition_size, int rest_num)
{
	// First kernel launch
	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	int offset;
	int last_row = partition_size;
	if(blockIdx.x < rest_num) {
		offset = blockIdx.x * (partition_size+1) * ((k<<1)+1);
		last_row = partition_size+1;
	} else {
		offset = (blockIdx.x * partition_size+rest_num) * ((k<<1)+1);
		last_row = partition_size;
	}

	if(c == 1) {
		dA[r+k+offset] /= dA[k+offset];
	}
	__syncthreads();
	dA[c*(k<<1) + r+k+offset] -= dA[r+k+offset] * dA[c*(k<<1)+k+offset];
	__syncthreads();

	// Second kernel launch
	for(int i=1; i<last_row-k; i++) {
		offset += (k<<1)+1;
		if(c == 1) {
			dA[r+k+offset] /= dA[k+offset];
		}
		__syncthreads();
		dA[c*(k<<1) + r+k+offset] -= dA[r+k+offset] * dA[c*(k<<1)+k+offset];
		__syncthreads();
	}

	for(int i=k; i>1; i--) {
		if(r >= i || c >= i) return ;
		offset += (k<<1) + 1;
		if(c == 1) {
			dA[r+k+offset] /= dA[k+offset];
		}
		__syncthreads();
		dA[c*(k<<1) + r+k+offset] -= dA[r+k+offset] * dA[c*(k<<1)+k+offset];
		__syncthreads();
	}
}


template <typename T>
__global__ void
swBandLU(T *dA, int k, int partition_size, int rest_num)
{
	// Calculate the row and column within the sliding window.
	int r = threadIdx.x % k + 1;
	int c = threadIdx.x / k + 1;

	// Calculate the number of rows in this partition and initialize
	// the pivot location (i.e. the top-left corner of the sliding
	// window).
	int pivotIdx;
	int numRows;

	if (blockIdx.x < rest_num) {
		pivotIdx = blockIdx.x * (partition_size + 1) * ((k<<1)+1) + k;
		numRows = partition_size + 1;
	} else {
		pivotIdx = (blockIdx.x * partition_size + rest_num) * ((k<<1)+1) + k;
		numRows = partition_size;
	}


	__shared__ T L[32];

	// Slide the window until its bottom-right corner reaches the end of
	// the partition. At each iteration, the first column of threads is
	// responsible for calculating the entries in L under the pivot. Then
	// all threads update the entries to the right of the pivot column.
	for (int i = 0; i < numRows - k; i++) {
		if (c == 1) {
			L[r-1] = dA[pivotIdx + r] / dA[pivotIdx];
			dA[pivotIdx + r] = L[r-1];
		}
		__syncthreads();

		dA[c*(k<<1) + pivotIdx + r] -= L[r-1] * dA[c*(k<<1) + pivotIdx];
		__syncthreads();

		pivotIdx += (k<<1) + 1;
	}

	// Slide the window through the last (k-1) pivots, ensuring we only
	// work inside the current partition.
	for (int i = k; i > 1; i--) {
		if (r >= i || c >= i) return ;

		if (c == 1) {
			L[r-1] = dA[pivotIdx + r] / dA[pivotIdx];
			dA[pivotIdx + r] = L[r-1];
		}
		__syncthreads();

		dA[c*(k<<1) + pivotIdx + r] -= L[r-1] * dA[c*(k<<1) + pivotIdx];
		__syncthreads();
		
		pivotIdx += (k<<1) + 1;
	}
}



template <typename T>
__global__ void
bandLU_safe(T *dA, int k, int partition_size, int rest_num)
{
	// First kernel launch
	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	int offset;
	int last_row = partition_size;
	if(blockIdx.x < rest_num) {
		offset = blockIdx.x * (partition_size+1) * ((k<<1)+1);
		last_row = partition_size+1;
	} else {
		offset = (blockIdx.x * partition_size+rest_num) * ((k<<1)+1);
		last_row = partition_size;
	}

	__shared__ T sharedA;

	if (threadIdx.x == 0) {
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
		//sharedA = dA[k+offset];
		//if (sharedA == 0.0)
			//sharedA = dA[k+offset] = (T)BURST_VALUE;
	}
	__syncthreads();

	if(c == 1) {
		dA[r+k+offset] /= sharedA;
	}
	__syncthreads();
	dA[c*(k<<1) + r+k+offset] -= dA[r+k+offset] * dA[c*(k<<1)+k+offset];
	__syncthreads();

	// Second kernel launch
	for(int i=1; i<last_row-k; i++) {
		offset += (k<<1)+1;
		if (threadIdx.x == 0) {
			sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
			//sharedA = dA[k+offset];
			//if (sharedA == 0.0)
				//sharedA = dA[k+offset] = (T)BURST_VALUE;
		}
		__syncthreads();
		if(c == 1) {
			dA[r+k+offset] /= sharedA;
		}
		__syncthreads();
		dA[c*(k<<1) + r+k+offset] -= dA[r+k+offset] * dA[c*(k<<1)+k+offset];
		__syncthreads();
	}

	for(int i=k; i>1; i--) {
		if(r >= i || c >= i) return ;
		offset += (k<<1) + 1;
		if (threadIdx.x == 0) {
			sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
			//sharedA = dA[k+offset];
			//if (sharedA == 0.0)
				//sharedA = dA[k+offset] = (T)BURST_VALUE;
		}
		__syncthreads();
		if(c == 1) {
			dA[r+k+offset] /= sharedA;
		}
		__syncthreads();
		dA[c*(k<<1) + r+k+offset] -= dA[r+k+offset] * dA[c*(k<<1)+k+offset];
		__syncthreads();
	}
}

template <typename T>
__global__ void
bandLU_g32(T *dA, int k, int partition_size, int rest_num)
{
	// First kernel launch
	int two_k = (k<<1);
	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	int offset;
	if (blockIdx.x < rest_num)
		offset = blockIdx.x * (partition_size+1) * (two_k+1);
	else
		offset = (blockIdx.x * partition_size + rest_num) * (two_k+1);

	int last_row = partition_size;
	if(blockIdx.x < rest_num)
		last_row++;

	int k_square = k*k;
	int tid = threadIdx.x;
	//int k_offset_sum = k+offset;

	for(int ttid = tid; ttid < k; ttid+=blockDim.x) {
		r = ttid + 1;
		dA[r+k+offset] /= dA[k+offset];
		//dA[r+k_offset_sum] /= dA[k_offset_sum];
	}
	__syncthreads();
	for(int ttid = tid; ttid < k_square; ttid+=blockDim.x) {
		r = ttid % k + 1;
		c = ttid / k + 1;
		dA[c*two_k + r+k+offset] -= dA[r+k+offset] * dA[c*two_k+k+offset];
		//dA[c*two_k + r+k_offset_sum] -= dA[r+k_offset_sum] * dA[c*two_k+k_offset_sum];
	}
	__syncthreads();

	// Second kernel launch
	for(int i=1; i<last_row-k; i++) {
		offset += two_k+1;
		//k_offset_sum += two_k+1;
		for(int ttid = tid; ttid < k; ttid+=blockDim.x) {
			r = ttid + 1;
			dA[r+k+offset] /= dA[k+offset];
			//dA[r+k_offset_sum] /= dA[k_offset_sum];
		}
		__syncthreads();
		for(int ttid = tid; ttid < k_square; ttid+=blockDim.x) {
			r = ttid % k + 1;
			c = ttid / k + 1;
			dA[c*two_k + r+k+offset] -= dA[r+k+offset] * dA[c*two_k+k+offset];
			//dA[c*two_k + r+k_offset_sum] -= dA[r+k_offset_sum] * dA[c*two_k+k_offset_sum];
		}
		__syncthreads();
	}

	for(int i=k; i>1; i--) {
		int i_minus_1_square = (i-1)*(i-1);
		int i_minus_1 = i-1;
		if(tid >= i_minus_1_square) return;
		offset += two_k + 1;
		//k_offset_sum += two_k + 1;
		for(int ttid = tid; ttid < i_minus_1; ttid+=blockDim.x) {
			r = ttid + 1;
			dA[r+k+offset] /= dA[k+offset];
			//dA[r+k_offset_sum] /= dA[k_offset_sum];
		}
		__syncthreads();
		for(int ttid = tid; ttid < i_minus_1_square; ttid+=blockDim.x) {
			r = ttid % i_minus_1 + 1;
			c = ttid / i_minus_1 + 1;
			dA[c*two_k + r+k+offset] -= dA[r+k+offset] * dA[c*two_k+k+offset];
			//dA[c*two_k + r+k_offset_sum] -= dA[r+k_offset_sum] * dA[c*two_k+k_offset_sum];
		}
		__syncthreads();
	}
}

template <typename T>
__global__ void
bandLU_g32_safe(T *dA, int k, int partition_size, int rest_num)
{
	// First kernel launch
	int two_k = (k<<1);
	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	int offset;
	if (blockIdx.x < rest_num)
		offset = blockIdx.x * (partition_size+1) * (two_k+1);
	else
		offset = (blockIdx.x * partition_size + rest_num) * (two_k+1);

	int last_row = partition_size;
	if(blockIdx.x < rest_num)
		last_row++;

	int k_square = k*k;
	int tid = threadIdx.x;

	__shared__ T sharedA;

	if (threadIdx.x == 0) {
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
		//sharedA = dA[k+offset];
		//if (sharedA == 0.0)
			//sharedA = dA[k+offset] = (T)BURST_VALUE;
	}
	__syncthreads();

	for(int ttid = tid; ttid < k; ttid+=blockDim.x) {
		r = ttid + 1;
		dA[r+k+offset] /= sharedA;
	}
	__syncthreads();
	for(int ttid = tid; ttid < k_square; ttid+=blockDim.x) {
		r = ttid % k + 1;
		c = ttid / k + 1;
		dA[c*two_k + r+k+offset] -= dA[r+k+offset] * dA[c*two_k+k+offset];
	}
	__syncthreads();

	// Second kernel launch
	for(int i=1; i<last_row-k; i++) {
		offset += two_k+1;
		//k_offset_sum += two_k+1;
		if (threadIdx.x == 0) {
			sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
			//sharedA = dA[k+offset];
			//if (sharedA == 0.0)
				//sharedA = dA[k+offset] = (T)BURST_VALUE;
		}
		__syncthreads();
		for(int ttid = tid; ttid < k; ttid+=blockDim.x) {
			r = ttid + 1;
			dA[r+k+offset] /= sharedA;
		}
		__syncthreads();
		for(int ttid = tid; ttid < k_square; ttid+=blockDim.x) {
			r = ttid % k + 1;
			c = ttid / k + 1;
			dA[c*two_k + r+k+offset] -= dA[r+k+offset] * dA[c*two_k+k+offset];
		}
		__syncthreads();
	}

	for(int i=k; i>1; i--) {
		int i_minus_1_square = (i-1)*(i-1);
		int i_minus_1 = i-1;
		if(tid >= i_minus_1_square) return;
		offset += two_k + 1;
		if (threadIdx.x == 0) {
			sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
			//sharedA = dA[k+offset];
			//if (sharedA == 0.0)
				//sharedA = dA[k+offset] = (T)BURST_VALUE;
		}
		__syncthreads();
		for(int ttid = tid; ttid < i_minus_1; ttid+=blockDim.x) {
			r = ttid + 1;
			dA[r+k+offset] /= sharedA;
		}
		__syncthreads();
		for(int ttid = tid; ttid < i_minus_1_square; ttid+=blockDim.x) {
			r = ttid % i_minus_1 + 1;
			c = ttid / i_minus_1 + 1;
			dA[c*two_k + r+k+offset] -= dA[r+k+offset] * dA[c*two_k+k+offset];
		}
		__syncthreads();
	}
}

template <typename T>
__global__ void
bandUL_g32(T *dA, int k, int partition_size, int rest_num)
{
	// First kernel launch
	int two_k = (k<<1);

	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	int offset;
	if (blockIdx.x < rest_num)
		offset = (blockIdx.x * (partition_size+1)+partition_size) * (two_k+1);
	else
		offset = (blockIdx.x * partition_size + rest_num + partition_size - 1) * (two_k+1);

	int last_row = partition_size;
	if (blockIdx.x < rest_num)
		last_row++;

	int tid = threadIdx.x;
	int k_square = k*k;

	for(int ttid = tid; ttid < k; ttid+=blockDim.x) {
		r = ttid + 1;
		dA[-r+k+offset] /= dA[k+offset];
	}
	__syncthreads();
	for(int ttid = tid; ttid < k_square; ttid+=blockDim.x) {
		r = ttid % k + 1, c = ttid / k + 1;
		dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
	}
	__syncthreads();

	// Second kernel launch
	for(int i=1; i<last_row-k; i++) {
		offset -= two_k+1;
		for(int ttid = tid; ttid < k; ttid+=blockDim.x) {
			r = ttid + 1;
			dA[-r+k+offset] /= dA[k+offset];
		}
		__syncthreads();
		for(int ttid = tid; ttid < k_square; ttid+=blockDim.x) {
			r = ttid % k + 1, c = ttid / k + 1;
			dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
		}
		__syncthreads();
	}

	for(int i=k; i>1; i--) {
		int i_minus_1 = i-1;
		int i_minus_1_square = i_minus_1*i_minus_1;
		if(tid >= i_minus_1_square) return;

		offset -= two_k + 1;
		for(int ttid = tid; ttid < i_minus_1; ttid+=blockDim.x) {
			r = ttid + 1;
			dA[-r+k+offset] /= dA[k+offset];
		}
		__syncthreads();
		for(int ttid = tid; ttid < i_minus_1_square; ttid+=blockDim.x) {
			r = ttid % i_minus_1 + 1, c = ttid / i_minus_1 + 1;
			dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
		}
		__syncthreads();
	}
}

template <typename T>
__global__ void
bandUL_g32_safe(T *dA, int k, int partition_size, int rest_num)
{
	// First kernel launch
	int two_k = (k<<1);

	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	int offset;
	if (blockIdx.x < rest_num)
		offset = (blockIdx.x * (partition_size+1)+partition_size) * (two_k+1);
	else
		offset = (blockIdx.x * partition_size + rest_num + partition_size - 1) * (two_k+1);

	int last_row = partition_size;
	if (blockIdx.x < rest_num)
		last_row++;

	int tid = threadIdx.x;
	int k_square = k*k;

	__shared__ T sharedA;

	if (threadIdx.x == 0) {
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
		//sharedA = dA[k+offset];
		//if (sharedA == 0.0)
			//sharedA = dA[k+offset] = (T)BURST_VALUE;
	}
	__syncthreads();

	for(int ttid = tid; ttid < k; ttid+=blockDim.x) {
		r = ttid + 1;
		dA[-r+k+offset] /= sharedA;
	}
	__syncthreads();
	for(int ttid = tid; ttid < k_square; ttid+=blockDim.x) {
		r = ttid % k + 1, c = ttid / k + 1;
		dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
	}
	__syncthreads();

	// Second kernel launch
	for(int i=1; i<last_row-k; i++) {
		offset -= two_k+1;
		if (threadIdx.x == 0) {
			sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
			//sharedA = dA[k+offset];
			//if (sharedA == 0.0)
				//sharedA = dA[k+offset] = (T)BURST_VALUE;
		}
		__syncthreads();
		for(int ttid = tid; ttid < k; ttid+=blockDim.x) {
			r = ttid + 1;
			dA[-r+k+offset] /= sharedA;
		}
		__syncthreads();
		for(int ttid = tid; ttid < k_square; ttid+=blockDim.x) {
			r = ttid % k + 1, c = ttid / k + 1;
			dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
		}
		__syncthreads();
	}

	for(int i=k; i>1; i--) {
		int i_minus_1 = i-1;
		int i_minus_1_square = i_minus_1*i_minus_1;
		if(tid >= i_minus_1_square) return;

		offset -= two_k + 1;
		if (threadIdx.x == 0) {
			sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
			//sharedA = dA[k+offset];
			//if (sharedA == 0.0)
				//sharedA = dA[k+offset] = (T)BURST_VALUE;
		}
		__syncthreads();
		for(int ttid = tid; ttid < i_minus_1; ttid+=blockDim.x) {
			r = ttid + 1;
			dA[-r+k+offset] /= sharedA;
		}
		__syncthreads();
		for(int ttid = tid; ttid < i_minus_1_square; ttid+=blockDim.x) {
			r = ttid % i_minus_1 + 1, c = ttid / i_minus_1 + 1;
			dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
		}
		__syncthreads();
	}
}

template <typename T>
__global__ void
bandUL(T *dA, int k, int partition_size, int rest_num)
{
	// First kernel launch
	int two_k = 2*k;

	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	//int offset = ((bid+1) * partition_size-1) *(two_k+1);
	int offset, last_row;
	if (blockIdx.x < rest_num) {
		offset = (blockIdx.x * (partition_size+1) + partition_size)* (two_k + 1);
		last_row = partition_size + 1;
	} else {
		offset = (blockIdx.x * partition_size+rest_num + partition_size - 1) * ((k<<1)+1);
		last_row = partition_size;
	}

	if(c == 1) {
		dA[-r+k+offset] /= dA[k+offset];
	}
	__syncthreads();
	dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
	__syncthreads();

	// Second kernel launch
	for(int i=1; i<last_row-k; i++) {
		offset -= two_k+1;
		if(c == 1) {
			dA[-r+k+offset] /= dA[k+offset];
		}
		__syncthreads();
		dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
		__syncthreads();
	}

	for(int i=k; i>1; i--) {
		if(r >= i || c >= i) return ;
		offset -= two_k + 1;
		if(c == 1) {
			dA[-r+k+offset] /= dA[k+offset];
		}
		__syncthreads();
		dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
		__syncthreads();
	}
}





template <typename T>
__global__ void
swBandUL(T *dA, int k, int partition_size, int rest_num)
{
	// Calculate the row and column within the sliding window.
	int r = threadIdx.x % k + 1;
	int c = threadIdx.x / k + 1;

	// Calculate the number of rows in this partition and initialize
	// the pivot location (i.e. the bottom-rigth corner of the sliding
	// window).
	int pivotIdx;
	int numRows;

	if (blockIdx.x < rest_num) {
		pivotIdx = (blockIdx.x * (partition_size+1) + partition_size) * ((k<<1) + 1) + k;
		numRows = partition_size + 1;
	} else {
		pivotIdx = (blockIdx.x * partition_size + rest_num + partition_size - 1) * ((k<<1) + 1) + k;
		numRows = partition_size;
	}


	__shared__ T U[32];

	// Slide the window until its top-left corner reaches the beginning of
	// the partition. At each iteration, the first column of threads is
	// responsible for calculating the entries in U above the pivot. Then
	// all threads update the entries to the left of the pivot column.
	for (int i = 0; i < numRows - k; i++) {
		if (c == 1) {
			U[r-1] = dA[pivotIdx - r] / dA[pivotIdx];
			dA[pivotIdx - r] = U[r-1];
		}
		__syncthreads();

		dA[-c*(k<<1) + pivotIdx - r] -= U[r-1] * dA[-c*(k<<1) + pivotIdx];
		__syncthreads();

		pivotIdx -= (k<<1) + 1;
	}

	// Slide the window through the last (k-1) pivots, ensuring we only
	// work inside the current partition.
	for (int i = k; i > 1; i--) {
		if (r >= i || c >= i) return ;

		if(c == 1) {
			U[r-1] = dA[pivotIdx - r] / dA[pivotIdx];
			dA[pivotIdx - r] = U[r-1];
		}
		__syncthreads();

		dA[-c*(k<<1) + pivotIdx - r] -= U[r-1] * dA[-c*(k<<1) + pivotIdx];
		__syncthreads();

		pivotIdx -= (k<<1) + 1;
	}
}





template <typename T>
__global__ void
bandUL_safe(T *dA, int k, int partition_size, int rest_num)
{
	// First kernel launch
	int two_k = 2*k;

	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	//int offset = ((bid+1) * partition_size-1) *(two_k+1);
	int offset, last_row;
	if (blockIdx.x < rest_num) {
		offset = (blockIdx.x * (partition_size+1) + partition_size)* (two_k + 1);
		last_row = partition_size + 1;
	} else {
		offset = (blockIdx.x * partition_size+rest_num + partition_size - 1) * ((k<<1)+1);
		last_row = partition_size;
	}

	__shared__ T sharedA;
	if (threadIdx.x == 0) {
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
		//sharedA = dA[k+offset];
		//if (sharedA == 0.0)
			//sharedA = dA[k+offset] = (T)BURST_VALUE;
	}
	__syncthreads();

	if(c == 1) {
		dA[-r+k+offset] /= sharedA;
	}
	__syncthreads();
	dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
	__syncthreads();

	// Second kernel launch
	for(int i=1; i<last_row-k; i++) {
		offset -= two_k+1;
		if (threadIdx.x == 0) {
			//sharedA = dA[k+offset];
			sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
			//if (sharedA == 0.0)
				//sharedA = dA[k+offset] = (T)BURST_VALUE;
		}
		__syncthreads();
		if(c == 1) {
			dA[-r+k+offset] /= sharedA;
		}
		__syncthreads();
		dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
		__syncthreads();
	}

	for(int i=k; i>1; i--) {
		if(r >= i || c >= i) return ;
		offset -= two_k + 1;
		if (threadIdx.x == 0) {
			sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
			//sharedA = dA[k+offset];
			//if (sharedA == 0.0)
				//sharedA = dA[k+offset] = (T)BURST_VALUE;
		}
		__syncthreads();
		if(c == 1) {
			dA[-r+k+offset] /= sharedA;
		}
		__syncthreads();
		dA[-c*two_k - r+k+offset] -= dA[-r+k+offset] * dA[-c*two_k+k+offset];
		__syncthreads();
	}
}

template <typename T>
__global__ void
bandLU_critical_div(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1;
	int offset;
	if(blockIdx.x < rest_num)
		offset = (blockIdx.x * (partition_size+1) + start_row) * ((k<<1) + 1);
	else {
		start_row--;
		offset = (blockIdx.x * partition_size + rest_num + start_row) * ((k<<1) + 1);
	}

	dA[r+k+offset] /= dA[k+offset];
}

template <typename T>
__global__ void
bandLU_critical_div_onePart(T *dA, int start_row, int k) {
	int r = threadIdx.x + 1;
	int pivotIdx = start_row * ((k<<1) + 1) + k;

	dA[r+pivotIdx] /= dA[pivotIdx];
}

template <typename T>
__global__ void
bandLU_critical_div_safe(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1;
	int offset;
	if(blockIdx.x < rest_num)
		offset = (blockIdx.x * (partition_size+1) + start_row) * ((k<<1) + 1);
	else {
		start_row--;
		offset = (blockIdx.x * partition_size + rest_num + start_row) * ((k<<1) + 1);
	}

	__shared__ T sharedA;
	if (threadIdx.x == 0) {
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
		//sharedA = dA[k+offset];
		//if (sharedA == 0.0)
			//sharedA = dA[k+offset] = (T)BURST_VALUE;
	}
	__syncthreads();
	dA[r+k+offset] /= sharedA;
}

template <typename T>
__global__ void
bandLU_critical_div_onePart_safe(T *dA, int start_row, int k) {
	int r = threadIdx.x + 1;
	int pivotIdx = start_row * ((k<<1) + 1) + k;

	__shared__ T sharedA;
	if (threadIdx.x == 0)
		sharedA = boostValue(dA[pivotIdx], dA[pivotIdx], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
	__syncthreads();

	dA[r+pivotIdx] /= sharedA;
}


template <typename T>
__global__ void
bandLU_critical_sub(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1, c = blockIdx.x + 1;
	int offset;
	if(blockIdx.y < rest_num)
		offset = (blockIdx.y * (partition_size+1) + start_row) * ((k<<1) + 1);
	else {
		start_row --;
		offset = (blockIdx.y * partition_size + rest_num + start_row) * ((k<<1) + 1);
	}

	dA[c*(k<<1)+ r+k+offset] -= dA[r+k+offset] * dA[c*(k<<1)+k+offset];
}

template <typename T>
__global__ void
bandLU_critical_sub_onePart(T *dA, int start_row, int k)
{
	int r = threadIdx.x + 1, c = blockIdx.x + 1;
	int pivotIdx = start_row * ((k<<1) + 1) + k;

	dA[c*(k<<1)+ r+pivotIdx] -= dA[r+pivotIdx] * dA[c*(k<<1)+pivotIdx];
}

template <typename T>
__global__ void 
bandLU_critical_sub_div_onePart (T *dA, int start_row, int k, int last_next)
{
	int r = threadIdx.x + 1, c = blockIdx.x + 1;
	int pivotIdx = start_row * ((k<<1) + 1) + k;

	dA[c*(k<<1)+ r+pivotIdx] -= dA[r+pivotIdx] * dA[c*(k<<1)+pivotIdx];

	if (c == 1) {
		pivotIdx += ((k<<1) + 1);
		__syncthreads();
		for (; r <= last_next; r += blockDim.x)
			dA[pivotIdx + r] /= dA[pivotIdx];
	}
}

template <typename T>
__global__ void 
bandLU_critical_sub_div_onePart_safe (T *dA, int start_row, int k, int last_next)
{
	int r = threadIdx.x + 1, c = blockIdx.x + 1;
	int pivotIdx = start_row * ((k<<1) + 1) + k;

	dA[c*(k<<1)+ r+pivotIdx] -= dA[r+pivotIdx] * dA[c*(k<<1)+pivotIdx];

	if (c == 1) {
		pivotIdx += ((k<<1) + 1);
		__shared__ T sharedA;
		if (threadIdx.x == 0) 
			sharedA = boostValue(dA[pivotIdx], dA[pivotIdx], (T)BURST_VALUE, (T)BURST_NEW_VALUE);

		__syncthreads();

		for (; r <= last_next; r += blockDim.x)
			dA[pivotIdx + r] /= sharedA;
	}
}

template <typename T>
__global__ void
bandLU_critical_div_general(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1;
	int bid = blockIdx.x;
	int offset;
	int last = k;
	if (bid < rest_num) {
		offset = (bid * (partition_size+1) + start_row) * ((k<<1) + 1);
		if (last > partition_size - start_row)
			last = partition_size - start_row;
	}
	else {
		start_row--;
		offset = (bid * partition_size + rest_num + start_row) * ((k<<1) + 1);
		if (last > partition_size - start_row - 1)
			last = partition_size - start_row - 1;
	}
	for(;r<=last; r+=blockDim.x)
		dA[r+k+offset] /= dA[k+offset];
}

template <typename T>
__global__ void
bandLU_critical_div_onePart_general(T *dA, int start_row, int k, int last) {
	int pivotIdx = start_row * ((k<<1) + 1) + k;
	int idx = threadIdx.x + blockIdx.x * blockDim.x + 1;
	if (idx > last)
		return;

	dA[pivotIdx + idx] /= dA[pivotIdx];
}

template <typename T>
__global__ void
bandLU_critical_div_safe_general(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1;
	int bid = blockIdx.x;
	int offset;
	int last = k;
	if (bid < rest_num) {
		offset = (bid * (partition_size+1) + start_row) * ((k<<1) + 1);
		if (last > partition_size - start_row)
			last = partition_size - start_row;
	}
	else {
		start_row--;
		offset = (bid * partition_size + rest_num + start_row) * ((k<<1) + 1);
		if (last > partition_size - start_row - 1)
			last = partition_size - start_row - 1;
	}
	__shared__ T sharedA;
	if (threadIdx.x == 0) {
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
		//sharedA = dA[k+offset];
		//if (sharedA == 0.0)
			//sharedA = dA[k+offset] = (T)BURST_VALUE;
	}
	__syncthreads();
	for(;r<=last; r+=blockDim.x)
		dA[r+k+offset] /= sharedA;
}

template <typename T>
__global__ void
bandLU_critical_div_onePart_safe_general(T *dA, int start_row, int k, int last) {
	__shared__ T sharedA;
	int pivotIdx = start_row * ((k<<1) + 1) + k;
	if (threadIdx.x == 0)
		sharedA = boostValue(dA[pivotIdx], dA[pivotIdx], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
	__syncthreads();

	for (int r = threadIdx.x + 1; r <= last; r+=blockDim.x)
		dA[pivotIdx+r] /= sharedA;
}

template <typename T>
__global__ void
bandLU_critical_sub_general(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1, c = blockIdx.x + 1;
	int bid = blockIdx.y;
	int offset;
	int last = k;
	if (bid < rest_num) {
		offset = (bid * (partition_size+1) + start_row) * ((k<<1) + 1);
		if (last > partition_size - start_row)
			last = partition_size - start_row;
	}
	else {
		start_row --;
		offset = (bid * partition_size + rest_num + start_row) * ((k<<1) + 1);
		if (last > partition_size - start_row - 1)
			last = partition_size - start_row - 1;
	}
	for(;r<=last; r+=blockDim.x)
		dA[c*(k<<1)+ r+k+offset] -= dA[r+k+offset] * dA[c*(k<<1)+k+offset];
}

template <typename T>
__global__ void
bandLU_critical_sub_onePart_general(T *dA, int start_row, int k, int last)
{
	int  c = blockIdx.x + 1;
	int pivotIdx = start_row * ((k<<1) + 1) + k;
	T tmp = dA[c*(k<<1)+pivotIdx];
	for(int r=threadIdx.x + 1;r<=last; r+=blockDim.x)
		dA[c*(k<<1)+ r+pivotIdx] -= dA[r+pivotIdx] * tmp;
}

template <typename T>
__global__ void
bandLU_critical_sub_div_onePart_general(T *dA, int start_row, int k, int last, int last_next)
{
	int  c = blockIdx.x + 1;
	int pivotIdx = start_row * ((k<<1) + 1) + k;
	T tmp = dA[c*(k<<1)+pivotIdx];
	for(int r=threadIdx.x + 1;r<=last; r+=blockDim.x)
		dA[c*(k<<1)+ r+pivotIdx] -= dA[r+pivotIdx] * tmp;

	if (c == 1) {
		pivotIdx += ((k<<1) + 1);
		__syncthreads();
		for (int r = threadIdx.x + 1; r <= last_next; r += blockDim.x)
			dA[pivotIdx + r] /= dA[pivotIdx];
	}
}

template <typename T>
__global__ void
bandLU_critical_sub_div_onePart_safe_general(T *dA, int start_row, int k, int last, int last_next)
{
	int  c = blockIdx.x + 1;
	int pivotIdx = start_row * ((k<<1) + 1) + k;
	T tmp = dA[c*(k<<1)+pivotIdx];
	for(int r=threadIdx.x + 1;r<=last; r+=blockDim.x)
		dA[c*(k<<1)+ r+pivotIdx] -= dA[r+pivotIdx] * tmp;

	if (c == 1) {
		pivotIdx += ((k<<1) + 1);
		__shared__ T sharedA;
		if (threadIdx.x == 0) 
			sharedA = boostValue(dA[pivotIdx], dA[pivotIdx], (T)BURST_VALUE, (T)BURST_NEW_VALUE);

		__syncthreads();
		for (int r = threadIdx.x + 1; r <= last_next; r += blockDim.x)
			dA[pivotIdx + r] /= sharedA;
	}
}

template <typename T>
__global__ void
blockedBandLU_critical_phase1(T *dA, int start_row, int k, int *last, int b)
{
	int pivotIdx;
	int last_row = start_row + b;

	pivotIdx = start_row * ((k<<1) + 1) + k;
	for (int row = start_row; row < last_row;  row++) {

		int cur_last = last[row];

		for (int tid = threadIdx.x + 1; tid <= cur_last; tid += blockDim.x)
			dA[pivotIdx + tid] /= dA[pivotIdx];

		__syncthreads();

		if (row == last_row - 1) break;
		if (cur_last == 0) {
			pivotIdx += ((k<<1) + 1);
			continue;
		}

		int num_elements = (last_row - row - 1) * cur_last;

		for (int tid = threadIdx.x; tid < num_elements; tid += blockDim.x) {
			int r = tid / cur_last + 1;
			int c = tid % cur_last + 1;

			dA[pivotIdx + c + r * (k << 1)] -= dA[pivotIdx + c] * dA[pivotIdx + r * (k << 1)];
		}

		__syncthreads();

		pivotIdx += ((k<<1) + 1);
	}
}

template <typename T>
__global__ void
blockedBandLU_critical_phase1_safe(T *dA, int start_row, int k, int *last, int b)
{
	int pivotIdx;
	int last_row = start_row + b;

	pivotIdx = start_row * ((k<<1) + 1) + k;

	__shared__ T sharedA;

	for (int row = start_row; row < last_row;  row++) {

		int cur_last = last[row];

		if (threadIdx.x == 0)
			sharedA = boostValue(dA[pivotIdx], dA[pivotIdx], (T)BURST_VALUE, (T)BURST_NEW_VALUE);

		__syncthreads();

		for (int tid = threadIdx.x + 1; tid <= cur_last; tid += blockDim.x)
			dA[pivotIdx + tid] /= sharedA;

		__syncthreads();

		if (row == last_row - 1) break;
		if (cur_last == 0) {
			pivotIdx += ((k<<1) + 1);
			continue;
		}

		int num_elements = (last_row - row - 1) * cur_last;

		for (int tid = threadIdx.x; tid < num_elements; tid += blockDim.x) {
			int r = tid / cur_last + 1;
			int c = tid % cur_last + 1;

			dA[pivotIdx + c + r * (k << 1)] -= dA[pivotIdx + c] * dA[pivotIdx + r * (k << 1)];
		}

		__syncthreads();

		pivotIdx += ((k<<1) + 1);
	}
}

template <typename T>
__global__ void
blockedBandLU_critical_phase2(T *dA, int start_row, int k, int b)
{
	int bid = blockIdx.x + b;
	int pivotIdx = start_row * ((k<<1) + 1) + k;

	extern __shared__ T sharedElem[];

	if (threadIdx.x + k < bid) {
		sharedElem[threadIdx.x] = (T)0;
		return;
	} else
		sharedElem[threadIdx.x] = dA[pivotIdx + bid * (k << 1) + threadIdx.x];

	__syncthreads();

	for (int i = 1; i < b; i++) {
		if (threadIdx.x >= i)
			sharedElem[threadIdx.x] -= sharedElem[i-1] * dA[pivotIdx + (i-1) * (k<<1) + threadIdx.x];

		__syncthreads();
	}

	dA[pivotIdx + bid * (k << 1) + threadIdx.x] = sharedElem[threadIdx.x];
}

template <typename T>
__global__ void
blockedBandLU_critical_phase3(T *dA, int start_row, int k, int cur_last, int b)
{
	int pivotIdx = start_row * ((k<<1) + 1) + k;
	int bid = blockIdx.x;

	for (int tid = threadIdx.x; tid < cur_last; tid += blockDim.x) {
		T tmp = dA[pivotIdx + b * ((k<<1)+1) + tid + (k<<1) * bid];
		for (int i = 0; i < b; i++)
			if (tid - i + b <= k && i + k >= b + bid)
				tmp -= dA[pivotIdx + tid + i * (k << 1) + b] * dA[pivotIdx + (b+bid) * (k << 1) + i];

		dA[pivotIdx + b * ((k<<1)+1) + tid + (k<<1) * bid] = tmp;
	}
}

template <typename T>
__global__ void
bandUL_critical_div(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1;
	int offset;
	if(blockIdx.x < rest_num)
		offset = (blockIdx.x * (partition_size+1) + start_row) * ((k<<1) + 1);
	else
		offset = (blockIdx.x * partition_size + rest_num + start_row) * ((k<<1) + 1);
	dA[-r+k+offset] /= dA[k+offset];
}

template <typename T>
__global__ void
bandUL_critical_div_safe(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1;
	int offset;
	if(blockIdx.x < rest_num)
		offset = (blockIdx.x * (partition_size+1) + start_row) * ((k<<1) + 1);
	else
		offset = (blockIdx.x * partition_size + rest_num + start_row) * ((k<<1) + 1);
	__shared__ T sharedA;
	if (threadIdx.x == 0) {
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
	}
	__syncthreads();
	dA[-r+k+offset] /= sharedA;
}

template <typename T>
__global__ void
bandUL_critical_sub(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1, c = blockIdx.x + 1;
	int offset;
	if(blockIdx.y < rest_num)
		offset = (blockIdx.y * (partition_size+1) + start_row) * ((k<<1) + 1);
	else
		offset = (blockIdx.y * partition_size + rest_num + start_row) * ((k<<1) + 1);

	dA[-c*(k<<1)- r+k+offset] -= dA[-r+k+offset] * dA[-c*(k<<1)+k+offset];
}

template <typename T>
__global__ void
bandUL_critical_div_general(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1;
	int offset;
	if(blockIdx.x < rest_num)
		offset = (blockIdx.x * (partition_size+1) + start_row) * ((k<<1) + 1);
	else
		offset = (blockIdx.x * partition_size + rest_num + start_row) * ((k<<1) + 1);
	int last = k;
	if(start_row < k)
		last = start_row;
	last++;
	for(; r<last; r+=blockDim.x)
		dA[-r+k+offset] /= dA[k+offset];
}

template <typename T>
__global__ void
bandUL_critical_div_safe_general(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1;
	int offset;
	if(blockIdx.x < rest_num)
		offset = (blockIdx.x * (partition_size+1) + start_row) * ((k<<1) + 1);
	else
		offset = (blockIdx.x * partition_size + rest_num + start_row) * ((k<<1) + 1);
	int last = k;
	if(start_row < k)
		last = start_row;
	last++;
	__shared__ T sharedA;
	if (threadIdx.x == 0)
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);

	__syncthreads();
	for(; r<last; r+=blockDim.x)
		dA[-r+k+offset] /= sharedA;
}

template <typename T>
__global__ void
bandUL_critical_sub_general(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int r = threadIdx.x + 1, c = blockIdx.x + 1;
	int offset;
	if (blockIdx.y < rest_num)
		offset = (blockIdx.y * (partition_size+1) + start_row) * ((k<<1)+ 1);
	else
		offset = (blockIdx.y * partition_size + rest_num + start_row) * ((k<<1)+ 1);
	int last = k;
	if(start_row < k)
		last = start_row;
	last++;
	for(; r<last; r+=blockDim.x)
		dA[-c*(k<<1)- r+k+offset] -= dA[-r+k+offset] * dA[-c*(k<<1)+k+offset];
}

// ============================================================
// This function follows bandLU to do division to matrix U,
// Currently works for k <= 1024 only
// ============================================================
template <typename T>
__global__ void
bandLU_post_divide(T *dA, int k, int N)
{
	int c = threadIdx.x, r = blockIdx.x + blockIdx.y * gridDim.x;
	if (r >= N || r + c - k < 0) return;
	dA[((k<<1)+1)*r + c] /= dA[((k<<1)+1)*(r+c-k) + k];
}

// ============================================================
// This function follows bandLU to do division to matrix U,
// This works for a general case
// ============================================================
template <typename T>
__global__ void
bandLU_post_divide_general(T *dA, int k, int N)
{
	int r = blockIdx.x + blockIdx.y * gridDim.x;
	if (r >= N) return;
	for (int c = threadIdx.x+k-blockDim.x; c>=0 && c>=k-r; c-= blockDim.x)
		dA[((k<<1)+1)*r + c] /= dA[((k<<1)+1)*(r+c-k) + k];
}

// ----------------------------------------------------------------------------
// CUDA kernels for LU factorization of a block-diagonal matrix with full
// diagonal blocks.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
fullLU_div(T *dA, int partition_size, int cur_row)
{
	int tid = threadIdx.x, bid = blockIdx.x;
	int offset = bid * partition_size * partition_size;
	__shared__ T sharedA;
	if(tid == 0)
		sharedA = dA[partition_size * cur_row + cur_row + offset];

	__syncthreads();
	dA[partition_size*cur_row + cur_row + tid + 1 + offset] /= sharedA;
}

template <typename T>
__global__ void
fullLU_div_safe(T *dA, int partition_size, int cur_row)
{
	int tid = threadIdx.x, bid = blockIdx.x;
	int offset = bid * partition_size * partition_size;
	__shared__ T sharedA;
	if(tid == 0) {
		sharedA = boostValue(dA[partition_size * cur_row + cur_row + offset], dA[partition_size * cur_row + cur_row + offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
	}
	__syncthreads();
	dA[partition_size*cur_row + cur_row + tid + 1 + offset] /= sharedA;
}

template <typename T>
__global__ void
fullLU_div_general(T *dA, int k, int partition_size, int cur_row)
{
	int tid = threadIdx.x;
	int offset = blockIdx.x * partition_size * partition_size;
	__shared__ T sharedA;
	if(tid == 0)
		sharedA = dA[partition_size * cur_row + cur_row + offset];
	__syncthreads();
	int it_last = 2*k-1-cur_row;
	for(;tid<it_last;tid+=blockDim.x)
		dA[partition_size*cur_row + cur_row + tid + 1 + offset] /= sharedA;
}

template <typename T>
__global__ void
fullLU_div_safe_general(T *dA, int k, int partition_size, int cur_row)
{
	int tid = threadIdx.x;
	int offset = blockIdx.x * partition_size * partition_size;
	__shared__ T sharedA;
	if(tid == 0)
		sharedA = boostValue(dA[partition_size * cur_row + cur_row + offset], dA[partition_size * cur_row + cur_row + offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);

	__syncthreads();
	int it_last = 2*k-1-cur_row;
	for(;tid<it_last;tid+=blockDim.x)
		dA[partition_size*cur_row + cur_row + tid + 1 + offset] /= sharedA;
}

template <typename T>
__global__ void
fullLU_sub(T *dA, int partition_size, int cur_row)
{
	int c = threadIdx.x + cur_row + 1, r = blockIdx.x + cur_row + 1;
	int offset = blockIdx.y * partition_size * partition_size;

	dA[partition_size*r + c+offset] -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];
}

template <typename T>
__global__ void
fullLU_sub_general(T *dA, int k, int partition_size, int cur_row)
{
	int c = threadIdx.x + cur_row + 1, r = blockIdx.x + cur_row + 1;
	int offset = blockIdx.y * partition_size * partition_size;

	int it_last = 2*k-1-cur_row;

	for(int tid = threadIdx.x;tid<it_last;tid+=blockDim.x, c+=blockDim.x) {
		dA[partition_size*r + c+offset] -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];
	}
}


template <typename T>
__global__ void
fullLU_sub_spec(T *dA, int partition_size, int k)
{
	int c = threadIdx.x+k, r = blockIdx.x+k, bidy = blockIdx.y;
	int offset = bidy * partition_size * partition_size;

	T tmp = dA[partition_size*r + c + offset];
	for (int cur_row = 0; cur_row < k; cur_row++)
		tmp -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];
	dA[partition_size*r + c + offset] = tmp;
}

template <typename T>
__global__ void
fullLU_sub_spec_general(T *dA, int partition_size, int k)
{
	int tid = threadIdx.x;
	int c = tid+k, r = blockIdx.x+k;
	int offset = blockIdx.y * partition_size * partition_size;
	for (int cur_row = 0; cur_row < k; cur_row++)
		for(; tid<k; tid+=blockDim.x, c+=blockDim.x)
			dA[partition_size*r + c +offset] -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];
}

template <typename T>
__global__ void
boostLastPivot(T *dA, int start_row, int k, int partition_size, int rest_num)
{
	int offset;
	if(blockIdx.x < rest_num)
		offset = (blockIdx.x * (partition_size+1) + start_row) * ((k<<1) + 1);
	else {
		start_row--;
		offset = (blockIdx.x * partition_size + rest_num + start_row) * ((k<<1) + 1);
	}
	boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
}



} // namespace device
} // namespace spike


#endif

