// ============================================================================
// This file contains the various matrix factorization CUDA kernels used for
// partitions of varying bandwidths.
// ============================================================================

#ifndef FACTOR_BAND_VAR_H
#define FACTOR_BAND_VAR_H

#include <cuda.h>


namespace spike {
namespace device {
namespace var {


template<typename T>
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

template<typename T>
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
		sharedA = boostValue(dA[k+offset], dA[k+offset], BURST_VALUE);
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
			sharedA = boostValue(dA[k+offset], dA[k+offset], BURST_VALUE);
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
			sharedA = boostValue(dA[k+offset], dA[k+offset], BURST_VALUE);
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

template<typename T>
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

template<typename T>
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
		sharedA = boostValue(dA[k+offset], dA[k+offset], BURST_VALUE);
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
			sharedA = boostValue(dA[k+offset], dA[k+offset], BURST_VALUE);
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
			sharedA = boostValue(dA[k+offset], dA[k+offset], BURST_VALUE);
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
		sharedA = boostValue(dA[k+offset], dA[k+offset], BURST_VALUE);
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


template<typename T>
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

template<typename T>
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
		sharedA = boostValue(dA[offset], dA[offset], BURST_VALUE);
	}
	__syncthreads();
	dA[tid + 1 + offset] /= sharedA;
}

template<typename T>
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

template<typename T>
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
		sharedA = boostValue(dA[offset], dA[offset], BURST_VALUE);
	__syncthreads();
	for(;tid<it_last;tid+=blockDim.x)
		dA[tid + 1 + offset] /= sharedA;
}

template<typename T>
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

template<typename T>
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


template<typename T>
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

template<typename T>
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

template<typename T>
__global__ void
fullLU_post_divide(T *dA, int *ks, int *offsets) {
	int k = ks[blockIdx.y];
	if (threadIdx.x > blockIdx.x)
		return;
	int offset = offsets[blockIdx.y];

	dA[offset + (k<<1)*(blockIdx.x+k+1) + threadIdx.x + k] /= dA[offset + (k<<1)*(threadIdx.x+k) + threadIdx.x + k];
}

template<typename T>
__global__ void
fullLU_post_divide_general(T *dA, int *ks, int *offsets) {
		int k = ks[blockIdx.y];
	if (threadIdx.x > blockIdx.x)
		return;
	int offset = offsets[blockIdx.y];

	for (int tid = threadIdx.x; tid <= blockIdx.x; tid += blockDim.x)
		dA[offset + (k<<1)*(blockIdx.x+k+1) + tid + k] /= dA[offset + (k<<1)*(tid+k) + tid + k];
}


} // namespace var
} // namespace device
} // namespace spike


#endif

