/** \file factor_band_var.cuh
 *  Various matrix factorization CUDA kernels used for the case of partitions
 *  with varying bandwidths.
 */

#ifndef FACTOR_BAND_VAR_H
#define FACTOR_BAND_VAR_H

#include <cuda.h>
#include <spike/common.h>

/** \namespace spike::device
 * \brief spike::device contains all CUDA kernels.
 */

/** \namespace spike::device::var
 * \brief spike::device::var contains all CUDA kernels for the
          variable bandwidth preconditioner.
 */

namespace spike {
namespace device {
namespace var {


template <typename T>
__global__ void
bandLU(T *dA, int *ks, int *offsets, int partition_size, int rest_num)
{
	// First kernel launch
	int k = ks[blockIdx.x];
	if (threadIdx.x >= k*k)
		return;

	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	int offset = offsets[blockIdx.x];
	int last_row = partition_size;
	if(blockIdx.x < rest_num) {
		last_row++;
	}

	if(c == 1)
		dA[r+k+offset] /= dA[k+offset];
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
bandLU_safe(T *dA, int *ks, int *offsets, int partition_size, int rest_num)
{
	// First kernel launch
	int k = ks[blockIdx.x];
	if (threadIdx.x >= k*k) return;
	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	int offset = offsets[blockIdx.x];
	int last_row = partition_size;
	if(blockIdx.x < rest_num) {
		last_row = partition_size++;
	}

	__shared__ T sharedA;

	if (threadIdx.x == 0) {
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
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
bandLU_g32(T *dA, int *ks, int *offsets, int partition_size, int rest_num)
{
	// First kernel launch
	int k = ks[blockIdx.x];
	if (threadIdx.x >= k*k) return;
	int two_k = (k<<1);
	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	int offset = offsets[blockIdx.x];

	int last_row = partition_size;
	if(blockIdx.x < rest_num)
		last_row++;

	int k_square = k*k;
	int tid = threadIdx.x;

	for(int ttid = tid; ttid < k; ttid+=blockDim.x) {
		r = ttid + 1;
		dA[r+k+offset] /= dA[k+offset];
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
		for(int ttid = tid; ttid < k; ttid+=blockDim.x) {
			r = ttid + 1;
			dA[r+k+offset] /= dA[k+offset];
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
		for(int ttid = tid; ttid < i_minus_1; ttid+=blockDim.x) {
			r = ttid + 1;
			dA[r+k+offset] /= dA[k+offset];
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
bandLU_g32_safe(T *dA, int *ks, int *offsets, int partition_size, int rest_num)
{
	// First kernel launch
	int k = ks[blockIdx.x];
	if (threadIdx.x >= k*k) return;
	int two_k = (k<<1);
	int r = threadIdx.x % k + 1, c = threadIdx.x / k+1;
	int offset = offsets[blockIdx.x];

	int last_row = partition_size;
	if(blockIdx.x < rest_num)
		last_row++;

	int k_square = k*k;
	int tid = threadIdx.x;

	__shared__ T sharedA;

	if (threadIdx.x == 0) {
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
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
		if (threadIdx.x == 0) {
			sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
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
bandLU_critical_div_general(T *dA, int start_row, int *ks, int *offsets, int partition_size, int rest_num)
{
	int k = ks[blockIdx.x];
	int r = threadIdx.x + 1;
	if (r > k) return;
	int bid = blockIdx.x;
	int offset = offsets[bid];
	int last = k;
	if (bid < rest_num) {
		offset += (start_row) * ((k<<1) + 1);
		if (last > partition_size - start_row)
			last = partition_size - start_row;
	}
	else {
		start_row--;
		offset += (start_row) * ((k<<1) + 1);
		if (last > partition_size - start_row - 1)
			last = partition_size - start_row - 1;
	}
	for(;r<=last; r+=blockDim.x)
		dA[r+k+offset] /= dA[k+offset];
}

template <typename T>
__global__ void
bandLU_critical_div_safe_general(T *dA, int start_row, int *ks, int *offsets, int partition_size, int rest_num)
{
	int k = ks[blockIdx.x];
	int r = threadIdx.x + 1;
	if (r > k) return;
	int bid = blockIdx.x;
	int offset = offsets[bid];
	int last = k;
	if (bid < rest_num) {
		offset += (start_row) * ((k<<1) + 1);
		if (last > partition_size - start_row)
			last = partition_size - start_row;
	}
	else {
		start_row--;
		offset += (start_row) * ((k<<1) + 1);
		if (last > partition_size - start_row - 1)
			last = partition_size - start_row - 1;
	}
	__shared__ T sharedA;
	if (threadIdx.x == 0) {
		sharedA = boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
	}
	__syncthreads();
	for(;r<=last; r+=blockDim.x)
		dA[r+k+offset] /= sharedA;
}

template <typename T>
__global__ void
bandLU_critical_sub_general(T *dA, int start_row, int *ks, int *offsets, int partition_size, int rest_num, int last)
{
	int k = ks[blockIdx.y];
	int r = threadIdx.x + 1, c = blockIdx.x + 1;
	if (r > k || c > k) return;
	int bid = blockIdx.y;
	int pivotIdx = offsets[bid] + k;
	//int last = k;
	if (last > k)
		last = k;
	if (last > partition_size - start_row)
		last = partition_size - start_row;
	if (bid >= rest_num)
		start_row --;
	pivotIdx += (start_row) * ((k<<1) + 1);

	T tmp = dA[c*(k<<1)+pivotIdx];
	for(;r<=last; r+=blockDim.x)
		dA[c*(k<<1)+ (r+pivotIdx)] -= dA[r+pivotIdx] * tmp;
}

template <typename T>
__global__ void
bandLU_critical_sub_div_general(T *dA, int start_row, int *ks, int *offsets, int partition_size, int rest_num, int *ks_col, int *ks_row)
{
	int k = ks[blockIdx.y];
	int r = threadIdx.x + 1, c = blockIdx.x + 1;
	if (r > k || c > k) return;
	int bid = blockIdx.y;
	int pivotIdx = offsets[bid] + k;

	int currow = start_row + partition_size * bid;
	if (bid < rest_num)
		currow += bid;
	else
		currow += rest_num - 1;

	if (c > ks_row[currow] && c != 1) return;

	int last = ks_col[currow];
	if (last > k)
		last = k;
	if (last > partition_size - start_row)
		last = partition_size - start_row;
	if (bid >= rest_num)
		start_row --;
	pivotIdx += (start_row) * ((k<<1) + 1);

	T tmp = dA[c*(k<<1)+pivotIdx];
	for(;r<=last; r+=blockDim.x)
		dA[c*(k<<1)+ (r+pivotIdx)] -= dA[r+pivotIdx] * tmp;

	if (c == 1) {
		start_row++;
		currow++;
		if (bid < rest_num) {
			if (start_row > partition_size)
				return;
		} else {
			if (start_row >= partition_size)
				return;
		}
		last = ks_col[currow];
		pivotIdx += ((k<<1) + 1);

		__syncthreads();
		for (r = threadIdx.x + 1; r <= last; r += blockDim.x)
			dA[r+pivotIdx] /= dA[pivotIdx];
	}
}

template <typename T>
__global__ void
bandLU_critical_sub_div_safe_general(T *dA, int start_row, int *ks, int *offsets, int partition_size, int rest_num, int *ks_col, int *ks_row)
{
	int k = ks[blockIdx.y];
	int r = threadIdx.x + 1, c = blockIdx.x + 1;
	if (r > k || c > k) return;
	int bid = blockIdx.y;
	int pivotIdx = offsets[bid] + k;

	int currow = start_row + partition_size * bid;
	if (bid < rest_num)
		currow += bid;
	else
		currow += rest_num - 1;

	if (c > ks_row[currow] && c != 1) return;

	int last = ks_col[currow];
	if (last > k)
		last = k;
	if (last > partition_size - start_row)
		last = partition_size - start_row;
	if (bid >= rest_num)
		start_row --;
	pivotIdx += (start_row) * ((k<<1) + 1);

	T tmp = dA[c*(k<<1)+pivotIdx];
	for(;r<=last; r+=blockDim.x)
		dA[c*(k<<1)+ (r+pivotIdx)] -= dA[r+pivotIdx] * tmp;

	if (c == 1) {
		start_row++;
		currow++;
		if (bid < rest_num) {
			if (start_row > partition_size)
				return;
		} else {
			if (start_row >= partition_size)
				return;
		}
		last = ks_col[currow];
		pivotIdx += ((k<<1) + 1);

		__shared__ T sharedA;
		if (threadIdx.x == 0)
			sharedA = boostValue(dA[pivotIdx], dA[pivotIdx], (T)BURST_VALUE, (T)BURST_NEW_VALUE);

		__syncthreads();
		for (r = threadIdx.x + 1; r <= last; r += blockDim.x)
			dA[r+pivotIdx] /= sharedA;
	}
}

// ============================================================
// This function follows bandLU to do division to matrix U,
// Currently works for k <= 1024 only
// ============================================================
template <typename T>
__global__ void
bandLU_post_divide_per_partition(T *dA, int k, int offset, int partSize)
{
	int c = threadIdx.x, r = blockIdx.x + blockIdx.y * gridDim.x;
	if (r >= partSize || r + c - k < 0) return;
	dA[offset + ((k<<1)+1)*r + c] /= dA[offset + ((k<<1)+1)*(r+c-k) + k];
}

// ============================================================
// This function follows bandLU to do division to matrix U,
// Currently works for general K
// ============================================================
template <typename T>
__global__ void
bandLU_post_divide_per_partition_general(T *dA, int k, int offset, int partSize)
{
	int r = blockIdx.x + blockIdx.y * gridDim.x;
	if (r >= partSize) return;
	for (int c = threadIdx.x + k - blockDim.x; c>=0 && c>=k-r; c-=blockDim.x)
		dA[offset + ((k<<1)+1)*r + c] /= dA[offset + ((k<<1)+1)*(r+c-k) + k];
}


template <typename T>
__global__ void
fullLU_div(T *dA, int *ks, int *offsets, int cur_row)
{
	int tid = threadIdx.x, bid = blockIdx.x;
	int k = ks[bid];
	int partition_size = (2*k);
	if (tid >= partition_size-1-cur_row) return;
	int offset = offsets[bid] + cur_row * partition_size + cur_row;
	__shared__ T sharedA;
	if(tid == 0) {
		sharedA = dA[offset];
	}
	__syncthreads();
	dA[tid + 1 + offset] /= sharedA;
}

template <typename T>
__global__ void
fullLU_div_safe(T *dA, int *ks, int *offsets, int cur_row)
{
	int tid = threadIdx.x, bid = blockIdx.x;
	int k = ks[bid];
	int partition_size = (2*k);
	if (tid >= partition_size-1-cur_row) return;
	int offset = offsets[bid] + cur_row * partition_size + cur_row;
	__shared__ T sharedA;
	if(tid == 0) {
		sharedA = boostValue(dA[offset], dA[offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
	}
	__syncthreads();
	dA[tid + 1 + offset] /= sharedA;
}

template <typename T>
__global__ void
fullLU_div_general(T *dA, int *ks, int *offsets, int cur_row)
{
	int tid = threadIdx.x;
	int k = ks[blockIdx.x];
	int it_last = 2*k-1-cur_row;
	if (tid >= it_last) return;

	int partition_size = (2*k);
	int offset = offsets[blockIdx.x] + partition_size * cur_row + cur_row;

	__shared__ T sharedA;
	if(tid == 0)
		sharedA = dA[offset];
	__syncthreads();
	for(;tid<it_last;tid+=blockDim.x)
		dA[tid + 1 + offset] /= sharedA;
}

template <typename T>
__global__ void
fullLU_div_safe_general(T *dA, int *ks, int *offsets, int cur_row)
{
	int tid = threadIdx.x;
	int k = ks[blockIdx.x];
	int it_last = 2*k-1-cur_row;
	if (tid >= it_last) return;

	int partition_size = (2*k);
	int offset = offsets[blockIdx.x] + partition_size * cur_row + cur_row;

	__shared__ T sharedA;
	if(tid == 0)
		sharedA = boostValue(dA[offset], dA[offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
	__syncthreads();
	for(;tid<it_last;tid+=blockDim.x)
		dA[tid + 1 + offset] /= sharedA;
}

template <typename T>
__global__ void
fullLU_sub(T *dA, int *ks, int *offsets, int cur_row)
{
	int k = ks[blockIdx.y];
	int partition_size = (2*k);
	int c = threadIdx.x + cur_row + 1, r = blockIdx.x + cur_row + 1;
	if (r >= partition_size || c >= partition_size) return;

	int offset = offsets[blockIdx.y];

	dA[partition_size*r + c+offset] -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];
}

template <typename T>
__global__ void
fullLU_sub_general(T *dA, int *ks, int *offsets, int cur_row)
{
	int k = ks[blockIdx.y];
	int it_last = 2*k-1-cur_row;
	if (threadIdx.x >= it_last || blockIdx.x >= it_last) return;
	int c = threadIdx.x + cur_row + 1, r = blockIdx.x + cur_row + 1;
	int partition_size = 2*k;
	int offset = offsets[blockIdx.y];

	for(int tid = threadIdx.x;tid<it_last;tid+=blockDim.x, c+=blockDim.x) {
		dA[partition_size*r + c+offset] -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];
	}
}

template <typename T>
__global__ void
fullLU_sub_div(T *dA, int *ks, int *offsets, int cur_row)
{
	int k = ks[blockIdx.y];
	int partition_size = (2*k);
	int c = threadIdx.x + cur_row + 1, r = blockIdx.x + cur_row + 1;
	if (r >= partition_size || c >= partition_size) return;

	int offset = offsets[blockIdx.y];

	dA[partition_size*r + c+offset] -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];

	if (blockIdx.x == 0) {
		cur_row++;
		if (c == partition_size - 1) return;
		__syncthreads();
		dA[partition_size*cur_row + cur_row + threadIdx.x + 1 + offset] /= dA[partition_size * cur_row + cur_row + offset];
	}
}

template <typename T>
__global__ void
fullLU_sub_div_general(T *dA, int *ks, int *offsets, int cur_row)
{
	int k = ks[blockIdx.y];
	int it_last = 2*k-1-cur_row;
	if (threadIdx.x >= it_last || blockIdx.x >= it_last) return;
	int c = threadIdx.x + cur_row + 1, r = blockIdx.x + cur_row + 1;
	int partition_size = 2*k;
	int offset = offsets[blockIdx.y];

	for(int tid = threadIdx.x;tid<it_last;tid+=blockDim.x, c+=blockDim.x)
		dA[partition_size*r + c+offset] -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];

	if (blockIdx.x == 0) {
		cur_row++;
		__syncthreads();
		T tmp = dA[partition_size * cur_row + cur_row + offset];
		for(int tid = threadIdx.x + 1; tid < it_last; tid+=blockDim.x)
			dA[partition_size*cur_row + cur_row + tid + offset] /= tmp;
	}
}

template <typename T>
__global__ void
fullLU_sub_div_safe(T *dA, int *ks, int *offsets, int cur_row)
{
	int k = ks[blockIdx.y];
	int partition_size = (2*k);
	int c = threadIdx.x + cur_row + 1, r = blockIdx.x + cur_row + 1;
	if (r >= partition_size || c >= partition_size) return;

	int offset = offsets[blockIdx.y];

	dA[partition_size*r + c+offset] -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];

	if (blockIdx.x == 0) {
		cur_row++;
		if (c == partition_size - 1) return;

		__shared__ T sharedA;
		if (threadIdx.x == 0)
			sharedA = boostValue(dA[partition_size * cur_row + cur_row + offset], dA[partition_size * cur_row + cur_row +offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
		__syncthreads();
		dA[partition_size*cur_row + cur_row + threadIdx.x + 1 + offset] /= sharedA;
	}
}

template <typename T>
__global__ void
fullLU_sub_div_safe_general(T *dA, int *ks, int *offsets, int cur_row)
{
	int k = ks[blockIdx.y];
	int it_last = 2*k-1-cur_row;
	if (threadIdx.x >= it_last || blockIdx.x >= it_last) return;
	int c = threadIdx.x + cur_row + 1, r = blockIdx.x + cur_row + 1;
	int partition_size = 2*k;
	int offset = offsets[blockIdx.y];

	for(int tid = threadIdx.x;tid<it_last;tid+=blockDim.x, c+=blockDim.x)
		dA[partition_size*r + c+offset] -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];

	if (blockIdx.x == 0) {
		cur_row++;
		__shared__ T sharedA;
		if (threadIdx.x == 0)
			sharedA = boostValue(dA[partition_size * cur_row + cur_row + offset], dA[partition_size * cur_row + cur_row +offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
		__syncthreads();
		for(int tid = threadIdx.x + 1; tid < it_last; tid+=blockDim.x)
			dA[partition_size*cur_row + cur_row + tid + offset] /= sharedA;
	}
}


template <typename T>
__global__ void
fullLU_sub_spec(T *dA, int *ks, int *offsets)
{
	int k = ks[blockIdx.y];
	if (threadIdx.x >= k || blockIdx.x >= k) return;
	int c = threadIdx.x+k, r = blockIdx.x+k, bidy = blockIdx.y;
	int partition_size = 2*k;
	int offset = offsets[bidy];

	T tmp = dA[partition_size*r+c+offset];
	for (int cur_row = 0; cur_row < k; cur_row++)
		tmp -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];
	dA[partition_size*r+c+offset] = tmp;
}

template <typename T>
__global__ void
fullLU_sub_spec_general(T *dA, int *ks, int *offsets)
{
	int k = ks[blockIdx.y];
	if (threadIdx.x >= k || blockIdx.x >= k) return;
	int tid = threadIdx.x;
	int c = tid+k, r = blockIdx.x+k;
	int partition_size = 2*k;
	int offset = offsets[blockIdx.y];

	for(tid = threadIdx.x; tid<k; tid+=blockDim.x, c+=blockDim.x)  {
		T tmp = dA[partition_size*r + c +offset];
		for (int cur_row = 0; cur_row < k; cur_row++)
			tmp -= dA[partition_size*r + cur_row+offset] * dA[partition_size*cur_row + c+offset];
		dA[partition_size*r + c +offset] = tmp;
	}
}

template <typename T>
__global__ void
fullLU_post_divide(T *dA, int *ks, int *offsets) {
	int k = ks[blockIdx.y];
	if (threadIdx.x > blockIdx.x)
		return;
	int offset = offsets[blockIdx.y];

	dA[offset + (k<<1)*(blockIdx.x+k+1) + threadIdx.x + k] /= dA[offset + (k<<1)*(threadIdx.x+k) + threadIdx.x + k];
}

template <typename T>
__global__ void
fullLU_post_divide_general(T *dA, int *ks, int *offsets) {
		int k = ks[blockIdx.y];
	if (threadIdx.x > blockIdx.x)
		return;
	int offset = offsets[blockIdx.y];

	for (int tid = threadIdx.x; tid <= blockIdx.x; tid += blockDim.x)
		dA[offset + (k<<1)*(blockIdx.x+k+1) + tid + k] /= dA[offset + (k<<1)*(tid+k) + tid + k];
}

template <typename T>
__global__ void
blockedFullLU_phase1_general(T *dA, int *ks, int *offsets, int cur_row, int b)
{
	int k = ks[blockIdx.x];
	int partition_size = (2*k);

	int offset = offsets[blockIdx.x] + cur_row * partition_size + cur_row;

	int last_row = cur_row + b;
	if (last_row > partition_size)
		last_row = partition_size;

	for (int i = 0; i < b; i++) {
		for (int tid = threadIdx.x + 1; tid < partition_size - cur_row; tid += blockDim.x)
			dA[tid + offset] /= dA[offset];

		__syncthreads();

		int element_count = (last_row - 1 - cur_row) * (partition_size - 1 - cur_row);

		if (element_count == 0) return;

		for (int tid = threadIdx.x; tid < element_count; tid += blockDim.x) {
			int r = tid / (partition_size - 1 - cur_row) + 1;
			int c = tid % (partition_size - 1 - cur_row) + 1;

			dA[offset + r * partition_size + c] -= dA[offset + r * partition_size] * dA[offset + c];
		}

		__syncthreads();

		offset += partition_size + 1;
		cur_row ++;
	}
}

template <typename T>
__global__ void
blockedFullLU_phase1_safe_general(T *dA, int *ks, int *offsets, int cur_row, int b)
{
	int k = ks[blockIdx.x];
	int partition_size = (2*k);

	int offset = offsets[blockIdx.x] + cur_row * partition_size + cur_row;

	int last_row = cur_row + b;
	if (last_row > partition_size)
		last_row = partition_size;

	__shared__ T sharedA;

	for (int i = 0; i < b; i++) {
		if (threadIdx.x == 0)
			sharedA = boostValue(dA[offset], dA[offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
		__syncthreads();

		for (int tid = threadIdx.x + 1; tid < partition_size - cur_row; tid += blockDim.x)
			dA[tid + offset] /= sharedA;

		__syncthreads();

		int element_count = (last_row - 1 - cur_row) * (partition_size - 1 - cur_row);

		if (element_count == 0) return;

		for (int tid = threadIdx.x; tid < element_count; tid += blockDim.x) {
			int r = tid / (partition_size - 1 - cur_row) + 1;
			int c = tid % (partition_size - 1 - cur_row) + 1;

			dA[offset + r * partition_size + c] -= dA[offset + r * partition_size] * dA[offset + c];
		}

		__syncthreads();

		offset += partition_size + 1;
		cur_row ++;
	}
}

template <typename T>
__global__ void
blockedFullLU_phase2_general(T *dA, int *ks, int *offsets, int cur_row, int b)
{
	int bid = blockIdx.x + b;
	int k = ks[blockIdx.y];

	int offset = offsets[blockIdx.y] + cur_row * k * 2 + cur_row;

	extern __shared__  T sharedElem[];

	sharedElem[threadIdx.x] = dA[offset + bid * k * 2 + threadIdx.x];

	__syncthreads();

	for (int i = 1; i < b; i++) {
		if (threadIdx.x >= i)
			sharedElem[threadIdx.x] -= sharedElem[i-1] * dA[offset + (i-1) * k * 2 + threadIdx.x];
		__syncthreads();
	}

	dA[offset + bid * k * 2 + threadIdx.x] = sharedElem[threadIdx.x];
}

template <typename T>
__global__ void
blockedFullLU_phase3_general(T *dA, int *ks, int *offsets, int cur_row, int b) {
	int k = ks[blockIdx.y];
	int partition_size = (k << 1);
	int offset = offsets[blockIdx.y] + cur_row * k * 2 + cur_row;

	int bid = blockIdx.x + b;

	for (int tid = threadIdx.x + b;  tid < partition_size - cur_row; tid += blockDim.x) {
		T tmp = dA[offset + bid * partition_size + tid];

		for (int i = 0; i < b; i++)
			tmp -= dA[offset + i * partition_size + tid] * dA[offset + bid * partition_size + i];

		dA[offset + bid * partition_size + tid] = tmp;
	}
}

template <typename T>
__global__ void
boostLastPivot(T *dA, int start_row, int *ks, int *offsets, int partition_size, int rest_num)
{
	int offset = offsets[blockIdx.x];
	int k = ks[blockIdx.x];
	if(blockIdx.x < rest_num)
		offset += start_row * ((k<<1) + 1);
	else {
		start_row--;
		offset += start_row * ((k<<1) + 1);
	}
	boostValue(dA[k+offset], dA[k+offset], (T)BURST_VALUE, (T)BURST_NEW_VALUE);
}


} // namespace var
} // namespace device
} // namespace spike


#endif

