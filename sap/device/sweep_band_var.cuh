/** \file factor_band_var.cuh
 *  Various forward/backward sweep CUDA kernels used for the case of partitions
 *  with varying bandwidths.
 */

#ifndef SWEEP_BAND_VAR_CUH
#define SWEEP_BAND_VAR_CUH

#include <cuda.h>


namespace sap {
namespace device {
namespace var {


// ----------------------------------------------------------------------------
// Kernels for forward and backward substitution using the LU decomposition of
// the truncated SPIKE reduced matrix.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
fwdElim_full_narrow(int N, int *ks, int *offsets, T *dA, T *dB, int b_partition_size, int b_rest_num)
{
	int bidx = blockIdx.x;
	int k = ks[bidx];
	int tid = threadIdx.x;
	int partition_size = (k<<1);
	int offset = offsets[bidx];

	if(bidx + 1 <= b_rest_num) {
		T tmp = dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)+tid];
		for(int i=0; i<k; i++) {
			 tmp -= dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i] * dA[i*partition_size + k+tid+offset];
		}
		dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)+tid] = tmp;
		__syncthreads();
		for(int i=k; i<partition_size-1; i++) {
			if(tid >= partition_size - i - 1) return;
			dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i+tid+1] -= dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i] * dA[i*partition_size + i+tid+1+offset];
			__syncthreads();
		}
	} else {
		T tmp = dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num+tid];
		for(int i=0; i<k; i++) {
			 tmp -= dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i] * dA[i*partition_size + k+tid+offset];
		}
		dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num+tid] = tmp;
		__syncthreads();
		for(int i=k; i<partition_size-1; i++) {
			if(tid >= partition_size - i - 1) return;
			dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i+tid+1] -= dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i] * dA[i*partition_size + i+tid+1+offset];
			__syncthreads();
		}
	}
}

template <typename T>
__global__ void
fwdElim_full(int N, int *ks, int *offsets, T *dA, T *dB, int b_partition_size, int b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x;
	int k = ks[bidx];
	int partition_size = (k<<1);
	int offset = offsets[bidx];

	int it_last = 2*k-1;

	if(bidx + 1 <= b_rest_num) {
		for (int ttid = tid+k; ttid <= it_last; ttid += blockDim.x) {
			T tmp = dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)+ttid-k];
			for(int i=0; i<k; i++) 
				tmp -= dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i] * dA[i*partition_size + ttid+offset];
			dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)+ttid-k] = tmp;
		}
		__syncthreads();
		for(int i=k; i<partition_size-1; i++) {
			if(tid >= partition_size - i - 1) return;
			if(it_last > partition_size - i - 1)
				it_last = partition_size - i-1;
			for(int ttid = tid; ttid <  it_last; ttid+=blockDim.x)
				dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i+ttid+1] -= dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i] * dA[i*partition_size + i + ttid+1+offset];
			__syncthreads();
		}
	} else {
		for (int ttid = tid+k; ttid <= it_last; ttid += blockDim.x) {
			T tmp = dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num+ttid-k];
			for(int i=0; i<k; i++)
				tmp -= dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i] * dA[i*partition_size + ttid+offset];
			dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num+ttid-k] = tmp;
		}
		__syncthreads();
		for(int i=k; i<partition_size-1; i++) {
			if(tid >= partition_size - i - 1) return;
			if(it_last > partition_size - i - 1)
				it_last = partition_size - i-1;
			for(int ttid = tid; ttid <  it_last; ttid+=blockDim.x)
				dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i+ttid+1] -= dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i] * dA[i*partition_size + i + ttid+1+offset];
			__syncthreads();
		}
	}
}

template <typename T>
__global__ void
preBck_full_divide_narrow(int N, int *ks, int *offsets, T *dA, T *dB, int b_partition_size, int b_rest_num)
{
	int k = ks[blockIdx.x];
	int offset = offsets[blockIdx.x];

	if (blockIdx.x < b_rest_num)
		dB[(blockIdx.x + 1) * (b_partition_size + 1) + threadIdx.x] /= dA[offset + (threadIdx.x+k)*(k<<1) + (threadIdx.x+k)];
	else
		dB[(blockIdx.x + 1) * b_partition_size + b_rest_num + threadIdx.x] /= dA[offset + (threadIdx.x+k)*(k<<1) + (threadIdx.x+k)];
}

template <typename T>
__global__ void
preBck_full_divide(int N, int *ks, int *offsets, T *dA, T *dB, int b_partition_size, int b_rest_num)
{
	int k = ks[blockIdx.x];
	int offset = offsets[blockIdx.x];

	if (blockIdx.x < b_rest_num) {
		for (int tid = threadIdx.x; tid < k; tid += blockDim.x)
			dB[(blockIdx.x + 1) * (b_partition_size + 1) + tid] /= dA[offset + (tid+k)*(k<<1) + (tid+k)];
	}
	else
		for (int tid = threadIdx.x; tid < k; tid += blockDim.x)
			dB[(blockIdx.x + 1) * b_partition_size + b_rest_num + tid] /= dA[offset + (tid+k)*(k<<1) + (tid+k)];
}

template <typename T>
__global__ void
bckElim_full_narrow(int N, int *ks, int *offsets, T *dA, T *dB, int b_partition_size, int b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x;
	int k = ks[bidx];
	int partition_size = (2*k);
	int offset = offsets[bidx];

	if(bidx+1 <= b_rest_num) {
		for(int i=partition_size-1; i>=k; i--) {
			if(tid >= i) return;

			dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i-tid-1] -= dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i] * dA[i*partition_size+i-tid-1+offset];
			__syncthreads();
		}
	} else {
		for(int i=partition_size-1; i>=k; i--) {
			if(tid >= i) return;

			dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i-tid-1] -= dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i] * dA[i*partition_size+i-tid-1+offset];
			__syncthreads();
		}
	}
}

template <typename T>
__global__ void
bckElim_full(int N, int *ks, int *offsets, T *dA, T *dB, int b_partition_size, int b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x;
	int k = ks[bidx];
	int partition_size = (2*k);
	int offset = offsets[bidx];

	int it_last = 2*k-1;

	if(bidx + 1 <= b_rest_num) {
		for(int i=partition_size-1; i>=k; i--) {
			if(tid >= i) return;

			if(it_last > i)
				it_last = i;
			for(int ttid = tid; ttid <  it_last; ttid+=blockDim.x)
				dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i-ttid-1] -= dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i] * dA[i*partition_size+i-ttid-1+offset];
			__syncthreads();
		}
	} else {
		for(int i=partition_size-1; i>=k; i--) {
			if(tid >= i) return;

			if(it_last > i)
				it_last = i;
			for(int ttid = tid; ttid <  it_last; ttid+=blockDim.x)
				dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i-ttid-1] -= dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i] * dA[i*partition_size+i-ttid-1+offset];
			__syncthreads();
		}
	}
}


// ----------------------------------------------------------------------------
// Kernels for forward and backward substitution 
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
fwdElim_sol(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int k = ks[blockIdx.x];
	if (tid >= k) return;
	int offset = offsets[blockIdx.x];
	int first_row = blockIdx.x*partition_size;
	int last_row;
	if(blockIdx.x < rest_num) {
		first_row += blockIdx.x;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	int it_last = k;

	for(int i=first_row; i<last_row-k; i++) {
		T tmp = dB[bidy*N+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i+ttid+1] -= tmp * dA[offset + k + ttid + 1];
		offset += (k<<1)+1;
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		if(it_last > last_row-i-1)
			it_last = last_row-i-1;
		T tmp = dB[bidy*N+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i+ttid+1] -= tmp * dA[offset + k + ttid + 1];
		offset += (k<<1)+1;
		__syncthreads();
	}
}

template <typename T>
__global__ void
fwdElimCholesky_sol(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int k = ks[blockIdx.x];
	if (tid >= k) return;
	int offset = offsets[blockIdx.x];
	int first_row = blockIdx.x*partition_size;
	int last_row;
	if(blockIdx.x < rest_num) {
		first_row += blockIdx.x;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	int it_last = k;

	for(int i=first_row; i<last_row-k; i++) {
		T tmp = dB[bidy*N+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i+ttid+1] -= tmp * dA[offset + ttid + 1];
		offset += k + 1;
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		if(it_last > last_row-i-1)
			it_last = last_row-i-1;
		T tmp = dB[bidy*N+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i+ttid+1] -= tmp * dA[offset + ttid + 1];
		offset += k+1;
		__syncthreads();
	}
}

template <typename T>
__global__ void
fwdElim_sol_medium(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int k = ks[bidx];
	if (tid >= k) return;
	int offset = offsets[bidx];
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}

	for(int i=first_row; i<last_row-k; i++) {
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[offset + k + tid + 1];
		offset += (k<<1)+1;
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[offset + k + tid + 1];
		offset += (k<<1)+1;
		__syncthreads();
	}
}

template <typename T>
__global__ void
fwdElimCholesky_sol_medium(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int k = ks[bidx];
	if (tid >= k) return;
	int offset = offsets[bidx];
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}

	for(int i=first_row; i<last_row-k; i++) {
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[offset + tid + 1];
		offset += k + 1;
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[offset + tid + 1];
		offset += k + 1;
		__syncthreads();
	}
}

template <typename T>
__global__ void
fwdElim_sol_narrow(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int k = ks[bidx];
	if (tid >= k) return;
	int offset = offsets[bidx];
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	for(int i=first_row; i<last_row-k; i++, offset += 2*k+1) {
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[offset + k + tid + 1];
	}
	for(int i=last_row-k; i<last_row-1; i++, offset += 2*k+1) {
		if(tid >= last_row-i-1) return;
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[offset + k + tid + 1];
	}
}

template <typename T>
__global__ void
fwdElimCholesky_sol_narrow(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int k = ks[bidx];
	if (tid >= k) return;
	int offset = offsets[bidx];
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	for(int i=first_row; i<last_row-k; i++, offset += k+1) {
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[offset + tid + 1];
	}
	for(int i=last_row-k; i<last_row-1; i++, offset += k+1) {
		if(tid >= last_row-i-1) return;
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[offset + tid + 1];
	}
}

template <typename T>
__global__ void
bckElim_sol(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int k = ks[blockIdx.x];
	if (tid >= k) return;

	int first_row = blockIdx.x*partition_size;
	int last_row;
	int offset;
	if(blockIdx.x < rest_num) {
		first_row += blockIdx.x;
		last_row = first_row + partition_size + 1;
		offset = offsets[blockIdx.x] + partition_size * ((k<<1)+1);
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
		offset = offsets[blockIdx.x] + (partition_size-1) * ((k<<1)+1);
	}

	int it_last = k;

	for(int i=last_row-1; i>=k+first_row; i--) {
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i-ttid-1] -= dB[bidy*N+i] * dA[offset+k-ttid-1];
		offset -= (k<<1)+1;
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;
		if(it_last > i-first_row)
			it_last = i-first_row;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i-ttid-1] -= dB[bidy*N+i] * dA[offset + k - ttid - 1];
		offset -= (k<<1)+1;
		__syncthreads();
	}
}

template <typename T>
__global__ void
bckElimCholesky_sol(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int k = ks[blockIdx.x];
	if (tid >= k) return;

	int first_row = blockIdx.x*partition_size;
	int last_row;
	int offset;
	if(blockIdx.x < rest_num) {
		first_row += blockIdx.x;
		last_row = first_row + partition_size + 1;
		offset = offsets[blockIdx.x] + partition_size * (k + 1);
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
		offset = offsets[blockIdx.x] + (partition_size-1) * (k + 1);
	}

	int it_last = k;

	for(int i=last_row-1; i>=k+first_row; i--) {
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i-ttid-1] -= dB[bidy*N+i] * dA[offset-(ttid+1) * k];
		offset -= k + 1;
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;
		if(it_last > i-first_row)
			it_last = i-first_row;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i-ttid-1] -= dB[bidy*N+i] * dA[offset - (ttid + 1) * k];
		offset -= k + 1;
		__syncthreads();
	}
}

template <typename T>
__global__ void
bckElim_sol_medium(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y * N;
	int k = ks[bidx];
	if (tid >= k) return;
	int pivotIdx;

	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
		pivotIdx = offsets[bidx] + partition_size * ((k<<1)+1) + k;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
		pivotIdx = offsets[bidx] + (partition_size-1) * ((k<<1)+1) + k;
	}

	for(int i=last_row-1; i>=k+first_row; i--) {
		dB[bidy+i-tid-1] -= dB[bidy + i] * dA[pivotIdx-tid-1];
		pivotIdx -= (k<<1)+1;
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;

		dB[bidy+i-tid-1] -= dB[bidy+i] * dA[pivotIdx - tid - 1];
		pivotIdx -= (k<<1)+1;
		__syncthreads();
	}
}

template <typename T>
__global__ void
bckElimCholesky_sol_medium(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y * N;
	int k = ks[bidx];
	if (tid >= k) return;
	int pivotIdx;

	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
		pivotIdx = offsets[bidx] + partition_size * (k + 1);
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
		pivotIdx = offsets[bidx] + (partition_size-1) * (k + 1);
	}

	for(int i=last_row-1; i>=k+first_row; i--) {
		dB[bidy+i-tid-1] -= dB[bidy + i] * dA[pivotIdx-(tid+1) * k];
		pivotIdx -= k+1;
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;

		dB[bidy+i-tid-1] -= dB[bidy+i] * dA[pivotIdx - (tid + 1) * k];
		pivotIdx -= k+1;
		__syncthreads();
	}
}

template <typename T>
__global__ void
bckElim_sol_narrow(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int k = ks[bidx];
	if (tid >= k) return;

	int first_row = bidx*partition_size;
	int last_row;
	int offset;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
		offset = offsets[bidx] + (partition_size) * ((k<<1)+1);
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
		offset = offsets[bidx] + (partition_size-1) * ((k<<1)+1);
	}

	for(int i=last_row-1; i>=k+first_row; i--) {
		dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[offset+k-tid-1];
		offset -= (k<<1) + 1;
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;

		dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[offset + k - tid - 1];
		offset -= (k<<1) + 1;
	}
}

template <typename T>
__global__ void
bckElimCholesky_sol_narrow(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int k = ks[bidx];
	if (tid >= k) return;

	int first_row = bidx*partition_size;
	int last_row;
	int offset;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
		offset = offsets[bidx] + (partition_size) * (k+1);
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
		offset = offsets[bidx] + (partition_size-1) * (k+1);
	}

	for(int i=last_row-1; i>=k+first_row; i--) {
		dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[offset-(tid+1) * k];
		offset -= k + 1;
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;

		dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[offset - (tid + 1) * k];
		offset -= k + 1;
	}
}

template <typename T>
__global__ void
preBck_sol_divide(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num, bool isSPD)
{
	int k = ks[blockIdx.y];
	int first_row = blockIdx.y*partition_size;
	int last_row;
	int pivotIdx = offsets[blockIdx.y] + (isSPD ? 0 : k);
	if(blockIdx.y < rest_num) {
		first_row += blockIdx.y;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= last_row-first_row)
		return;
	dB[first_row + idx] /= dA[pivotIdx + idx * (isSPD ? (k + 1) : ((k<<1)+1))];
}

template <typename T>
__device__ void
fwdElim_offDiag_large_tiled(T *dA, T *dB, int idx, int k, int g_k, int r, int first_row, int last_row, int offset, T *a_elements, int column_width) {
	int i;
	int step = blockDim.x;
	if (blockDim.x * (blockIdx.x + 1) > r)
		step = r - blockDim.x * blockIdx.x;
	for (i = first_row + 1; i + 20 <= last_row; i += 20) {
		T tmp0 = dB[g_k * i + idx];
		T tmp1 = dB[g_k * (i+1) + idx];
		T tmp2 = dB[g_k * (i+2) + idx];
		T tmp3 = dB[g_k * (i+3) + idx];
		T tmp4 = dB[g_k * (i+4) + idx];
		T tmp5 = dB[g_k * (i+5) + idx];
		T tmp6 = dB[g_k * (i+6) + idx];
		T tmp7 = dB[g_k * (i+7) + idx];
		T tmp8 = dB[g_k * (i+8) + idx];
		T tmp9 = dB[g_k * (i+9) + idx];
		T tmp10 = dB[g_k * (i+10) + idx];
		T tmp11 = dB[g_k * (i+11) + idx];
		T tmp12 = dB[g_k * (i+12) + idx];
		T tmp13 = dB[g_k * (i+13) + idx];
		T tmp14 = dB[g_k * (i+14) + idx];
		T tmp15 = dB[g_k * (i+15) + idx];
		T tmp16 = dB[g_k * (i+16) + idx];
		T tmp17 = dB[g_k * (i+17) + idx];
		T tmp18 = dB[g_k * (i+18) + idx];
		T tmp19 = dB[g_k * (i+19) + idx];

		int row_to_start;
		if (i - k < first_row)
			row_to_start = first_row;
		else
			row_to_start = i - k;

		const int MAX_COUNT = 1020;
		int counter = MAX_COUNT;

		for (int j = i - 1; j >= row_to_start; j--) {
			T tmp20 = dB[g_k * j + idx];
			if (counter == MAX_COUNT) {
				__syncthreads();
				for (int l = threadIdx.x; l < MAX_COUNT; l += step) {
					int row = l / 20;
					if (j - row < row_to_start)
						break;
					int col = l % 20;
					a_elements[l] = (i + col - j + row > k ? (T) 0 : dA[offset + column_width * (j - row - first_row) + (i + col) - j + row]);
				}
				counter = 0;
				__syncthreads();
			}
			/*
			tmp0 -= tmp20 * dA[offset + column_width * (j - first_row) + i - j];
			tmp1 -= ((i + 1 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+1) - j]);
			tmp2 -= ((i + 2 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+2) - j]);
			tmp3 -= ((i + 3 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+3) - j]);
			tmp4 -= ((i + 4 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+4) - j]);
			tmp5 -= ((i + 5 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+5) - j]);
			tmp6 -= ((i + 6 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+6) - j]);
			tmp7 -= ((i + 7 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+7) - j]);
			tmp8 -= ((i + 8 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+8) - j]);
			tmp9 -= ((i + 9 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+9) - j]);
			tmp10 -= ((i + 10 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+10) - j]);
			tmp11 -= ((i + 11 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+11) - j]);
			tmp12 -= ((i + 12 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+12) - j]);
			tmp13 -= ((i + 13 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+13) - j]);
			tmp14 -= ((i + 14 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+14) - j]);
			tmp15 -= ((i + 15 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+15) - j]);
			tmp16 -= ((i + 16 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+16) - j]);
			tmp17 -= ((i + 17 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+17) - j]);
			tmp18 -= ((i + 18 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+18) - j]);
			tmp19 -= ((i + 19 - j > k)? 0 : tmp20 * dA[offset + column_width * (j - first_row) + (i+19) - j]);
			*/
			tmp0 -= tmp20 * a_elements[counter];
			tmp1 -= ((i + 1 - j > k)? 0 : tmp20 * a_elements[counter + 1]);
			tmp2 -= ((i + 2 - j > k)? 0 : tmp20 * a_elements[counter + 2]);
			tmp3 -= ((i + 3 - j > k)? 0 : tmp20 * a_elements[counter + 3]);
			tmp4 -= ((i + 4 - j > k)? 0 : tmp20 * a_elements[counter + 4]);
			tmp5 -= ((i + 5 - j > k)? 0 : tmp20 * a_elements[counter + 5]);
			tmp6 -= ((i + 6 - j > k)? 0 : tmp20 * a_elements[counter + 6]);
			tmp7 -= ((i + 7 - j > k)? 0 : tmp20 * a_elements[counter + 7]);
			tmp8 -= ((i + 8 - j > k)? 0 : tmp20 * a_elements[counter + 8]);
			tmp9 -= ((i + 9 - j > k)? 0 : tmp20 * a_elements[counter + 9]);
			tmp10 -= ((i + 10 - j > k)? 0 : tmp20 * a_elements[counter + 10]);
			tmp11 -= ((i + 11 - j > k)? 0 : tmp20 * a_elements[counter + 11]);
			tmp12 -= ((i + 12 - j > k)? 0 : tmp20 * a_elements[counter + 12]);
			tmp13 -= ((i + 13 - j > k)? 0 : tmp20 * a_elements[counter + 13]);
			tmp14 -= ((i + 14 - j > k)? 0 : tmp20 * a_elements[counter + 14]);
			tmp15 -= ((i + 15 - j > k)? 0 : tmp20 * a_elements[counter + 15]);
			tmp16 -= ((i + 16 - j > k)? 0 : tmp20 * a_elements[counter + 16]);
			tmp17 -= ((i + 17 - j > k)? 0 : tmp20 * a_elements[counter + 17]);
			tmp18 -= ((i + 18 - j > k)? 0 : tmp20 * a_elements[counter + 18]);
			tmp19 -= ((i + 19 - j > k)? 0 : tmp20 * a_elements[counter + 19]);

			counter += 20;
		}
		__syncthreads();
		int cur_idx = 0;
		for (int loadCnt = 19; loadCnt > 0; loadCnt --) {
			for (int l = threadIdx.x + 1; l <= loadCnt; l += step)
				a_elements[cur_idx + l - 1] = dA[offset + column_width * (i + 19 - loadCnt - first_row) + l];

			cur_idx += loadCnt;
		}
		__syncthreads();

		tmp1 -= tmp0 * a_elements[0];
		tmp2 -= tmp0 * a_elements[1];
		tmp3 -= tmp0 * a_elements[2];
		tmp4 -= tmp0 * a_elements[3];
		tmp5 -= tmp0 * a_elements[4];
		tmp6 -= tmp0 * a_elements[5];
		tmp7 -= tmp0 * a_elements[6];
		tmp8 -= tmp0 * a_elements[7];
		tmp9 -= tmp0 * a_elements[8];
		tmp10 -= tmp0 * a_elements[9];
		tmp11 -= tmp0 * a_elements[10];
		tmp12 -= tmp0 * a_elements[11];
		tmp13 -= tmp0 * a_elements[12];
		tmp14 -= tmp0 * a_elements[13];
		tmp15 -= tmp0 * a_elements[14];
		tmp16 -= tmp0 * a_elements[15];
		tmp17 -= tmp0 * a_elements[16];
		tmp18 -= tmp0 * a_elements[17];
		tmp19 -= tmp0 * a_elements[18];
		tmp2 -= tmp1 * a_elements[19];
		tmp3 -= tmp1 * a_elements[20];
		tmp4 -= tmp1 * a_elements[21];
		tmp5 -= tmp1 * a_elements[22];
		tmp6 -= tmp1 * a_elements[23];
		tmp7 -= tmp1 * a_elements[24];
		tmp8 -= tmp1 * a_elements[25];
		tmp9 -= tmp1 * a_elements[26];
		tmp10 -= tmp1 * a_elements[27];
		tmp11 -= tmp1 * a_elements[28];
		tmp12 -= tmp1 * a_elements[29];
		tmp13 -= tmp1 * a_elements[30];
		tmp14 -= tmp1 * a_elements[31];
		tmp15 -= tmp1 * a_elements[32];
		tmp16 -= tmp1 * a_elements[33];
		tmp17 -= tmp1 * a_elements[34];
		tmp18 -= tmp1 * a_elements[35];
		tmp19 -= tmp1 * a_elements[36];
		tmp3 -= tmp2 * a_elements[37];
		tmp4 -= tmp2 * a_elements[38];
		tmp5 -= tmp2 * a_elements[39];
		tmp6 -= tmp2 * a_elements[40];
		tmp7 -= tmp2 * a_elements[41];
		tmp8 -= tmp2 * a_elements[42];
		tmp9 -= tmp2 * a_elements[43];
		tmp10 -= tmp2 * a_elements[44];
		tmp11 -= tmp2 * a_elements[45];
		tmp12 -= tmp2 * a_elements[46];
		tmp13 -= tmp2 * a_elements[47];
		tmp14 -= tmp2 * a_elements[48];
		tmp15 -= tmp2 * a_elements[49];
		tmp16 -= tmp2 * a_elements[50];
		tmp17 -= tmp2 * a_elements[51];
		tmp18 -= tmp2 * a_elements[52];
		tmp19 -= tmp2 * a_elements[53];
		tmp4 -= tmp3 * a_elements[54];
		tmp5 -= tmp3 * a_elements[55];
		tmp6 -= tmp3 * a_elements[56];
		tmp7 -= tmp3 * a_elements[57];
		tmp8 -= tmp3 * a_elements[58];
		tmp9 -= tmp3 * a_elements[59];
		tmp10 -= tmp3 * a_elements[60];
		tmp11 -= tmp3 * a_elements[61];
		tmp12 -= tmp3 * a_elements[62];
		tmp13 -= tmp3 * a_elements[63];
		tmp14 -= tmp3 * a_elements[64];
		tmp15 -= tmp3 * a_elements[65];
		tmp16 -= tmp3 * a_elements[66];
		tmp17 -= tmp3 * a_elements[67];
		tmp18 -= tmp3 * a_elements[68];
		tmp19 -= tmp3 * a_elements[69];
		tmp5 -= tmp4 * a_elements[70];
		tmp6 -= tmp4 * a_elements[71];
		tmp7 -= tmp4 * a_elements[72];
		tmp8 -= tmp4 * a_elements[73];
		tmp9 -= tmp4 * a_elements[74];
		tmp10 -= tmp4 * a_elements[75];
		tmp11 -= tmp4 * a_elements[76];
		tmp12 -= tmp4 * a_elements[77];
		tmp13 -= tmp4 * a_elements[78];
		tmp14 -= tmp4 * a_elements[79];
		tmp15 -= tmp4 * a_elements[80];
		tmp16 -= tmp4 * a_elements[81];
		tmp17 -= tmp4 * a_elements[82];
		tmp18 -= tmp4 * a_elements[83];
		tmp19 -= tmp4 * a_elements[84];
		tmp6 -= tmp5 * a_elements[85];
		tmp7 -= tmp5 * a_elements[86];
		tmp8 -= tmp5 * a_elements[87];
		tmp9 -= tmp5 * a_elements[88];
		tmp10 -= tmp5 * a_elements[89];
		tmp11 -= tmp5 * a_elements[90];
		tmp12 -= tmp5 * a_elements[91];
		tmp13 -= tmp5 * a_elements[92];
		tmp14 -= tmp5 * a_elements[93];
		tmp15 -= tmp5 * a_elements[94];
		tmp16 -= tmp5 * a_elements[95];
		tmp17 -= tmp5 * a_elements[96];
		tmp18 -= tmp5 * a_elements[97];
		tmp19 -= tmp5 * a_elements[98];
		tmp7 -= tmp6 * a_elements[99];
		tmp8 -= tmp6 * a_elements[100];
		tmp9 -= tmp6 * a_elements[101];
		tmp10 -= tmp6 * a_elements[102];
		tmp11 -= tmp6 * a_elements[103];
		tmp12 -= tmp6 * a_elements[104];
		tmp13 -= tmp6 * a_elements[105];
		tmp14 -= tmp6 * a_elements[106];
		tmp15 -= tmp6 * a_elements[107];
		tmp16 -= tmp6 * a_elements[108];
		tmp17 -= tmp6 * a_elements[109];
		tmp18 -= tmp6 * a_elements[110];
		tmp19 -= tmp6 * a_elements[111];
		tmp8 -= tmp7 * a_elements[112];
		tmp9 -= tmp7 * a_elements[113];
		tmp10 -= tmp7 * a_elements[114];
		tmp11 -= tmp7 * a_elements[115];
		tmp12 -= tmp7 * a_elements[116];
		tmp13 -= tmp7 * a_elements[117];
		tmp14 -= tmp7 * a_elements[118];
		tmp15 -= tmp7 * a_elements[119];
		tmp16 -= tmp7 * a_elements[120];
		tmp17 -= tmp7 * a_elements[121];
		tmp18 -= tmp7 * a_elements[122];
		tmp19 -= tmp7 * a_elements[123];
		tmp9 -= tmp8 * a_elements[124];
		tmp10 -= tmp8 * a_elements[125];
		tmp11 -= tmp8 * a_elements[126];
		tmp12 -= tmp8 * a_elements[127];
		tmp13 -= tmp8 * a_elements[128];
		tmp14 -= tmp8 * a_elements[129];
		tmp15 -= tmp8 * a_elements[130];
		tmp16 -= tmp8 * a_elements[131];
		tmp17 -= tmp8 * a_elements[132];
		tmp18 -= tmp8 * a_elements[133];
		tmp19 -= tmp8 * a_elements[134];
		tmp10 -= tmp9 * a_elements[135];
		tmp11 -= tmp9 * a_elements[136];
		tmp12 -= tmp9 * a_elements[137];
		tmp13 -= tmp9 * a_elements[138];
		tmp14 -= tmp9 * a_elements[139];
		tmp15 -= tmp9 * a_elements[140];
		tmp16 -= tmp9 * a_elements[141];
		tmp17 -= tmp9 * a_elements[142];
		tmp18 -= tmp9 * a_elements[143];
		tmp19 -= tmp9 * a_elements[144];
		tmp11 -= tmp10 * a_elements[145];
		tmp12 -= tmp10 * a_elements[146];
		tmp13 -= tmp10 * a_elements[147];
		tmp14 -= tmp10 * a_elements[148];
		tmp15 -= tmp10 * a_elements[149];
		tmp16 -= tmp10 * a_elements[150];
		tmp17 -= tmp10 * a_elements[151];
		tmp18 -= tmp10 * a_elements[152];
		tmp19 -= tmp10 * a_elements[153];
		tmp12 -= tmp11 * a_elements[154];
		tmp13 -= tmp11 * a_elements[155];
		tmp14 -= tmp11 * a_elements[156];
		tmp15 -= tmp11 * a_elements[157];
		tmp16 -= tmp11 * a_elements[158];
		tmp17 -= tmp11 * a_elements[159];
		tmp18 -= tmp11 * a_elements[160];
		tmp19 -= tmp11 * a_elements[161];
		tmp13 -= tmp12 * a_elements[162];
		tmp14 -= tmp12 * a_elements[163];
		tmp15 -= tmp12 * a_elements[164];
		tmp16 -= tmp12 * a_elements[165];
		tmp17 -= tmp12 * a_elements[166];
		tmp18 -= tmp12 * a_elements[167];
		tmp19 -= tmp12 * a_elements[168];
		tmp14 -= tmp13 * a_elements[169];
		tmp15 -= tmp13 * a_elements[170];
		tmp16 -= tmp13 * a_elements[171];
		tmp17 -= tmp13 * a_elements[172];
		tmp18 -= tmp13 * a_elements[173];
		tmp19 -= tmp13 * a_elements[174];
		tmp15 -= tmp14 * a_elements[175];
		tmp16 -= tmp14 * a_elements[176];
		tmp17 -= tmp14 * a_elements[177];
		tmp18 -= tmp14 * a_elements[178];
		tmp19 -= tmp14 * a_elements[179];
		tmp16 -= tmp15 * a_elements[180];
		tmp17 -= tmp15 * a_elements[181];
		tmp18 -= tmp15 * a_elements[182];
		tmp19 -= tmp15 * a_elements[183];
		tmp17 -= tmp16 * a_elements[184];
		tmp18 -= tmp16 * a_elements[185];
		tmp19 -= tmp16 * a_elements[186];
		tmp18 -= tmp17 * a_elements[187];
		tmp19 -= tmp17 * a_elements[188];
		tmp19 -= tmp18 * a_elements[189];

		dB[g_k * i + idx] = tmp0;
		dB[g_k * (i+1) + idx] = tmp1;
		dB[g_k * (i+2) + idx] = tmp2;
		dB[g_k * (i+3) + idx] = tmp3;
		dB[g_k * (i+4) + idx] = tmp4;
		dB[g_k * (i+5) + idx] = tmp5;
		dB[g_k * (i+6) + idx] = tmp6;
		dB[g_k * (i+7) + idx] = tmp7;
		dB[g_k * (i+8) + idx] = tmp8;
		dB[g_k * (i+9) + idx] = tmp9;
		dB[g_k * (i+10) + idx] = tmp10;
		dB[g_k * (i+11) + idx] = tmp11;
		dB[g_k * (i+12) + idx] = tmp12;
		dB[g_k * (i+13) + idx] = tmp13;
		dB[g_k * (i+14) + idx] = tmp14;
		dB[g_k * (i+15) + idx] = tmp15;
		dB[g_k * (i+16) + idx] = tmp16;
		dB[g_k * (i+17) + idx] = tmp17;
		dB[g_k * (i+18) + idx] = tmp18;
		dB[g_k * (i+19) + idx] = tmp19;
	}

	for (; i < last_row; i++) {
		T tmp = dB[g_k * i + idx];

		int row_to_start;
		if (i - k < first_row)
			row_to_start = first_row;
		else
			row_to_start = i - k;

		for (int j = i-1; j >= row_to_start; j--)
			tmp -= dB[g_k * j + idx] * dA[offset + column_width * (j - first_row) + i - j];

		dB[g_k * i + idx] = tmp;
	}
}

template <typename T>
__device__ void
bckElim_offDiag_large_tiled(T *dA, T *dB, int idx, int k, int g_k, int r, int first_row, int last_row, int offset, T *a_elements, int column_width, int factor) {
	int i;
	int step = blockDim.x;
	if (blockDim.x * (blockIdx.x + 1) > r)
		step = r - blockDim.x * blockIdx.x;

	for (i = last_row - 2; i - 20 >= first_row; i -= 20) {
		T tmp0 = dB[g_k * i + idx];
		T tmp1 = dB[g_k * (i-1) + idx];
		T tmp2 = dB[g_k * (i-2) + idx];
		T tmp3 = dB[g_k * (i-3) + idx];
		T tmp4 = dB[g_k * (i-4) + idx];
		T tmp5 = dB[g_k * (i-5) + idx];
		T tmp6 = dB[g_k * (i-6) + idx];
		T tmp7 = dB[g_k * (i-7) + idx];
		T tmp8 = dB[g_k * (i-8) + idx];
		T tmp9 = dB[g_k * (i-9) + idx];
		T tmp10 = dB[g_k * (i-10) + idx];
		T tmp11 = dB[g_k * (i-11) + idx];
		T tmp12 = dB[g_k * (i-12) + idx];
		T tmp13 = dB[g_k * (i-13) + idx];
		T tmp14 = dB[g_k * (i-14) + idx];
		T tmp15 = dB[g_k * (i-15) + idx];
		T tmp16 = dB[g_k * (i-16) + idx];
		T tmp17 = dB[g_k * (i-17) + idx];
		T tmp18 = dB[g_k * (i-18) + idx];
		T tmp19 = dB[g_k * (i-19) + idx];

		int row_to_start;
		if (i + k < last_row)
			row_to_start = i + k;
		else
			row_to_start = last_row - 1;

		const int MAX_COUNT = 1020;
		int counter = MAX_COUNT;
		for (int j = i + 1; j <= row_to_start; j++) {
			T tmp20 = dB[g_k * j + idx];

			if (counter == MAX_COUNT) {
				__syncthreads();

				for (int l = threadIdx.x; l < MAX_COUNT; l += step) {
					int row = l / 20;
					if (j + row > row_to_start) {
						break;
					}
					int col = l % 20;
					a_elements[l] = ((i - col - j - row < -k )? (T) 0 : dA[offset - column_width * (last_row - 1 - j - row) + (i - col - j - row) * factor]);

				}
				counter = 0;
				__syncthreads();
			}
			/*
			tmp0 -= tmp20 * dA[offset - column_width * (last_row - 1 - j) + i - j];
			tmp1 -= ((i - 1 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-1) - j]);
			tmp2 -= ((i - 2 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-2) - j]);
			tmp3 -= ((i - 3 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-3) - j]);
			tmp4 -= ((i - 4 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-4) - j]);
			tmp5 -= ((i - 5 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-5) - j]);
			tmp6 -= ((i - 6 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-6) - j]);
			tmp7 -= ((i - 7 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-7) - j]);
			tmp8 -= ((i - 8 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-8) - j]);
			tmp9 -= ((i - 9 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-9) - j]);
			tmp10 -= ((i - 10 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-10) - j]);
			tmp11 -= ((i - 11 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-11) - j]);
			tmp12 -= ((i - 12 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-12) - j]);
			tmp13 -= ((i - 13 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-13) - j]);
			tmp14 -= ((i - 14 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-14) - j]);
			tmp15 -= ((i - 15 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-15) - j]);
			tmp16 -= ((i - 16 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-16) - j]);
			tmp17 -= ((i - 17 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-17) - j]);
			tmp18 -= ((i - 18 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-18) - j]);
			tmp19 -= ((i - 19 - j < -k)? 0 : tmp20 * dA[offset - column_width * (last_row - 1 - j) + (i-19) - j]);
			*/
			tmp0 -= tmp20 * a_elements[counter];
			tmp1 -= ((i - 1 - j < -k)? 0 : tmp20 * a_elements[counter + 1]);
			tmp2 -= ((i - 2 - j < -k)? 0 : tmp20 * a_elements[counter + 2]);
			tmp3 -= ((i - 3 - j < -k)? 0 : tmp20 * a_elements[counter + 3]);
			tmp4 -= ((i - 4 - j < -k)? 0 : tmp20 * a_elements[counter + 4]);
			tmp5 -= ((i - 5 - j < -k)? 0 : tmp20 * a_elements[counter + 5]);
			tmp6 -= ((i - 6 - j < -k)? 0 : tmp20 * a_elements[counter + 6]);
			tmp7 -= ((i - 7 - j < -k)? 0 : tmp20 * a_elements[counter + 7]);
			tmp8 -= ((i - 8 - j < -k)? 0 : tmp20 * a_elements[counter + 8]);
			tmp9 -= ((i - 9 - j < -k)? 0 : tmp20 * a_elements[counter + 9]);
			tmp10 -= ((i - 10 - j < -k)? 0 : tmp20 * a_elements[counter + 10]);
			tmp11 -= ((i - 11 - j < -k)? 0 : tmp20 * a_elements[counter + 11]);
			tmp12 -= ((i - 12 - j < -k)? 0 : tmp20 * a_elements[counter + 12]);
			tmp13 -= ((i - 13 - j < -k)? 0 : tmp20 * a_elements[counter + 13]);
			tmp14 -= ((i - 14 - j < -k)? 0 : tmp20 * a_elements[counter + 14]);
			tmp15 -= ((i - 15 - j < -k)? 0 : tmp20 * a_elements[counter + 15]);
			tmp16 -= ((i - 16 - j < -k)? 0 : tmp20 * a_elements[counter + 16]);
			tmp17 -= ((i - 17 - j < -k)? 0 : tmp20 * a_elements[counter + 17]);
			tmp18 -= ((i - 18 - j < -k)? 0 : tmp20 * a_elements[counter + 18]);
			tmp19 -= ((i - 19 - j < -k)? 0 : tmp20 * a_elements[counter + 19]);

			counter += 20;
		}

		__syncthreads();

		int cur_idx = 0;
		for (int loadCnt = 19; loadCnt > 0; loadCnt --) {
			for (int l = threadIdx.x + 1; l <= loadCnt; l += step)
				a_elements[cur_idx + l - 1] = dA[offset - column_width * (last_row - i + 18 - loadCnt) - l * factor];
			cur_idx += loadCnt;
		}

		__syncthreads();

		tmp1 -= tmp0 * a_elements[0];
		tmp2 -= tmp0 * a_elements[1];
		tmp3 -= tmp0 * a_elements[2];
		tmp4 -= tmp0 * a_elements[3];
		tmp5 -= tmp0 * a_elements[4];
		tmp6 -= tmp0 * a_elements[5];
		tmp7 -= tmp0 * a_elements[6];
		tmp8 -= tmp0 * a_elements[7];
		tmp9 -= tmp0 * a_elements[8];
		tmp10 -= tmp0 * a_elements[9];
		tmp11 -= tmp0 * a_elements[10];
		tmp12 -= tmp0 * a_elements[11];
		tmp13 -= tmp0 * a_elements[12];
		tmp14 -= tmp0 * a_elements[13];
		tmp15 -= tmp0 * a_elements[14];
		tmp16 -= tmp0 * a_elements[15];
		tmp17 -= tmp0 * a_elements[16];
		tmp18 -= tmp0 * a_elements[17];
		tmp19 -= tmp0 * a_elements[18];
		tmp2 -= tmp1 * a_elements[19];
		tmp3 -= tmp1 * a_elements[20];
		tmp4 -= tmp1 * a_elements[21];
		tmp5 -= tmp1 * a_elements[22];
		tmp6 -= tmp1 * a_elements[23];
		tmp7 -= tmp1 * a_elements[24];
		tmp8 -= tmp1 * a_elements[25];
		tmp9 -= tmp1 * a_elements[26];
		tmp10 -= tmp1 * a_elements[27];
		tmp11 -= tmp1 * a_elements[28];
		tmp12 -= tmp1 * a_elements[29];
		tmp13 -= tmp1 * a_elements[30];
		tmp14 -= tmp1 * a_elements[31];
		tmp15 -= tmp1 * a_elements[32];
		tmp16 -= tmp1 * a_elements[33];
		tmp17 -= tmp1 * a_elements[34];
		tmp18 -= tmp1 * a_elements[35];
		tmp19 -= tmp1 * a_elements[36];
		tmp3 -= tmp2 * a_elements[37];
		tmp4 -= tmp2 * a_elements[38];
		tmp5 -= tmp2 * a_elements[39];
		tmp6 -= tmp2 * a_elements[40];
		tmp7 -= tmp2 * a_elements[41];
		tmp8 -= tmp2 * a_elements[42];
		tmp9 -= tmp2 * a_elements[43];
		tmp10 -= tmp2 * a_elements[44];
		tmp11 -= tmp2 * a_elements[45];
		tmp12 -= tmp2 * a_elements[46];
		tmp13 -= tmp2 * a_elements[47];
		tmp14 -= tmp2 * a_elements[48];
		tmp15 -= tmp2 * a_elements[49];
		tmp16 -= tmp2 * a_elements[50];
		tmp17 -= tmp2 * a_elements[51];
		tmp18 -= tmp2 * a_elements[52];
		tmp19 -= tmp2 * a_elements[53];
		tmp4 -= tmp3 * a_elements[54];
		tmp5 -= tmp3 * a_elements[55];
		tmp6 -= tmp3 * a_elements[56];
		tmp7 -= tmp3 * a_elements[57];
		tmp8 -= tmp3 * a_elements[58];
		tmp9 -= tmp3 * a_elements[59];
		tmp10 -= tmp3 * a_elements[60];
		tmp11 -= tmp3 * a_elements[61];
		tmp12 -= tmp3 * a_elements[62];
		tmp13 -= tmp3 * a_elements[63];
		tmp14 -= tmp3 * a_elements[64];
		tmp15 -= tmp3 * a_elements[65];
		tmp16 -= tmp3 * a_elements[66];
		tmp17 -= tmp3 * a_elements[67];
		tmp18 -= tmp3 * a_elements[68];
		tmp19 -= tmp3 * a_elements[69];
		tmp5 -= tmp4 * a_elements[70];
		tmp6 -= tmp4 * a_elements[71];
		tmp7 -= tmp4 * a_elements[72];
		tmp8 -= tmp4 * a_elements[73];
		tmp9 -= tmp4 * a_elements[74];
		tmp10 -= tmp4 * a_elements[75];
		tmp11 -= tmp4 * a_elements[76];
		tmp12 -= tmp4 * a_elements[77];
		tmp13 -= tmp4 * a_elements[78];
		tmp14 -= tmp4 * a_elements[79];
		tmp15 -= tmp4 * a_elements[80];
		tmp16 -= tmp4 * a_elements[81];
		tmp17 -= tmp4 * a_elements[82];
		tmp18 -= tmp4 * a_elements[83];
		tmp19 -= tmp4 * a_elements[84];
		tmp6 -= tmp5 * a_elements[85];
		tmp7 -= tmp5 * a_elements[86];
		tmp8 -= tmp5 * a_elements[87];
		tmp9 -= tmp5 * a_elements[88];
		tmp10 -= tmp5 * a_elements[89];
		tmp11 -= tmp5 * a_elements[90];
		tmp12 -= tmp5 * a_elements[91];
		tmp13 -= tmp5 * a_elements[92];
		tmp14 -= tmp5 * a_elements[93];
		tmp15 -= tmp5 * a_elements[94];
		tmp16 -= tmp5 * a_elements[95];
		tmp17 -= tmp5 * a_elements[96];
		tmp18 -= tmp5 * a_elements[97];
		tmp19 -= tmp5 * a_elements[98];
		tmp7 -= tmp6 * a_elements[99];
		tmp8 -= tmp6 * a_elements[100];
		tmp9 -= tmp6 * a_elements[101];
		tmp10 -= tmp6 * a_elements[102];
		tmp11 -= tmp6 * a_elements[103];
		tmp12 -= tmp6 * a_elements[104];
		tmp13 -= tmp6 * a_elements[105];
		tmp14 -= tmp6 * a_elements[106];
		tmp15 -= tmp6 * a_elements[107];
		tmp16 -= tmp6 * a_elements[108];
		tmp17 -= tmp6 * a_elements[109];
		tmp18 -= tmp6 * a_elements[110];
		tmp19 -= tmp6 * a_elements[111];
		tmp8 -= tmp7 * a_elements[112];
		tmp9 -= tmp7 * a_elements[113];
		tmp10 -= tmp7 * a_elements[114];
		tmp11 -= tmp7 * a_elements[115];
		tmp12 -= tmp7 * a_elements[116];
		tmp13 -= tmp7 * a_elements[117];
		tmp14 -= tmp7 * a_elements[118];
		tmp15 -= tmp7 * a_elements[119];
		tmp16 -= tmp7 * a_elements[120];
		tmp17 -= tmp7 * a_elements[121];
		tmp18 -= tmp7 * a_elements[122];
		tmp19 -= tmp7 * a_elements[123];
		tmp9 -= tmp8 * a_elements[124];
		tmp10 -= tmp8 * a_elements[125];
		tmp11 -= tmp8 * a_elements[126];
		tmp12 -= tmp8 * a_elements[127];
		tmp13 -= tmp8 * a_elements[128];
		tmp14 -= tmp8 * a_elements[129];
		tmp15 -= tmp8 * a_elements[130];
		tmp16 -= tmp8 * a_elements[131];
		tmp17 -= tmp8 * a_elements[132];
		tmp18 -= tmp8 * a_elements[133];
		tmp19 -= tmp8 * a_elements[134];
		tmp10 -= tmp9 * a_elements[135];
		tmp11 -= tmp9 * a_elements[136];
		tmp12 -= tmp9 * a_elements[137];
		tmp13 -= tmp9 * a_elements[138];
		tmp14 -= tmp9 * a_elements[139];
		tmp15 -= tmp9 * a_elements[140];
		tmp16 -= tmp9 * a_elements[141];
		tmp17 -= tmp9 * a_elements[142];
		tmp18 -= tmp9 * a_elements[143];
		tmp19 -= tmp9 * a_elements[144];
		tmp11 -= tmp10 * a_elements[145];
		tmp12 -= tmp10 * a_elements[146];
		tmp13 -= tmp10 * a_elements[147];
		tmp14 -= tmp10 * a_elements[148];
		tmp15 -= tmp10 * a_elements[149];
		tmp16 -= tmp10 * a_elements[150];
		tmp17 -= tmp10 * a_elements[151];
		tmp18 -= tmp10 * a_elements[152];
		tmp19 -= tmp10 * a_elements[153];
		tmp12 -= tmp11 * a_elements[154];
		tmp13 -= tmp11 * a_elements[155];
		tmp14 -= tmp11 * a_elements[156];
		tmp15 -= tmp11 * a_elements[157];
		tmp16 -= tmp11 * a_elements[158];
		tmp17 -= tmp11 * a_elements[159];
		tmp18 -= tmp11 * a_elements[160];
		tmp19 -= tmp11 * a_elements[161];
		tmp13 -= tmp12 * a_elements[162];
		tmp14 -= tmp12 * a_elements[163];
		tmp15 -= tmp12 * a_elements[164];
		tmp16 -= tmp12 * a_elements[165];
		tmp17 -= tmp12 * a_elements[166];
		tmp18 -= tmp12 * a_elements[167];
		tmp19 -= tmp12 * a_elements[168];
		tmp14 -= tmp13 * a_elements[169];
		tmp15 -= tmp13 * a_elements[170];
		tmp16 -= tmp13 * a_elements[171];
		tmp17 -= tmp13 * a_elements[172];
		tmp18 -= tmp13 * a_elements[173];
		tmp19 -= tmp13 * a_elements[174];
		tmp15 -= tmp14 * a_elements[175];
		tmp16 -= tmp14 * a_elements[176];
		tmp17 -= tmp14 * a_elements[177];
		tmp18 -= tmp14 * a_elements[178];
		tmp19 -= tmp14 * a_elements[179];
		tmp16 -= tmp15 * a_elements[180];
		tmp17 -= tmp15 * a_elements[181];
		tmp18 -= tmp15 * a_elements[182];
		tmp19 -= tmp15 * a_elements[183];
		tmp17 -= tmp16 * a_elements[184];
		tmp18 -= tmp16 * a_elements[185];
		tmp19 -= tmp16 * a_elements[186];
		tmp18 -= tmp17 * a_elements[187];
		tmp19 -= tmp17 * a_elements[188];
		tmp19 -= tmp18 * a_elements[189];

		dB[g_k * i + idx] = tmp0;
		dB[g_k * (i-1) + idx] = tmp1;
		dB[g_k * (i-2) + idx] = tmp2;
		dB[g_k * (i-3) + idx] = tmp3;
		dB[g_k * (i-4) + idx] = tmp4;
		dB[g_k * (i-5) + idx] = tmp5;
		dB[g_k * (i-6) + idx] = tmp6;
		dB[g_k * (i-7) + idx] = tmp7;
		dB[g_k * (i-8) + idx] = tmp8;
		dB[g_k * (i-9) + idx] = tmp9;
		dB[g_k * (i-10) + idx] = tmp10;
		dB[g_k * (i-11) + idx] = tmp11;
		dB[g_k * (i-12) + idx] = tmp12;
		dB[g_k * (i-13) + idx] = tmp13;
		dB[g_k * (i-14) + idx] = tmp14;
		dB[g_k * (i-15) + idx] = tmp15;
		dB[g_k * (i-16) + idx] = tmp16;
		dB[g_k * (i-17) + idx] = tmp17;
		dB[g_k * (i-18) + idx] = tmp18;
		dB[g_k * (i-19) + idx] = tmp19;
	}
	for (; i >= first_row; i--) {
		T tmp = dB[g_k * i + idx];

		int row_to_start;
		if (i + k < last_row)
			row_to_start = i + k;
		else
			row_to_start = last_row - 1;

		for (int j = i + 1; j <= row_to_start; j++) {
			tmp -= dB[g_k * j + idx] * dA[offset - column_width * (last_row - 1 - j) + (i - j) * factor];
		}

		dB[g_k * i + idx] = tmp;
	}
}

template <typename T>
__global__ void
fwdElim_spike(int N, int *ks, int g_k, int rightWidth, int *offsets, T *dA, T *dB, int partition_size, int rest_num, int *left_spike_widths, int *right_spike_widths, int *first_rows, bool isSPD, int right_count, int left_count, int left_offset)
{
	__shared__ T a_elements[1024];
	////__shared__ T a_elements[2];

	int k, offset, first_row, last_row, idx, bidy = blockIdx.y;
	int column_width;

	if (bidy < right_count) {
		k = ks[bidy];
		column_width = k + 1;
		if (!isSPD)
			column_width += k;

		offset = offsets[bidy] + (isSPD ? 0 : k);
		first_row = bidy*partition_size;
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= right_spike_widths[bidy]) return;
		if (bidy < rest_num) {
			first_row += bidy;
			last_row = first_row + partition_size + 1;
		} else {
			first_row += rest_num;
			last_row = first_row + partition_size;
		}
        if (first_rows != NULL) {
            offset += column_width * (first_rows[bidy] - first_row);
            first_row = first_rows[bidy];
        }
		fwdElim_offDiag_large_tiled(dA, dB, idx, k, g_k, right_spike_widths[bidy], first_row, last_row, offset, a_elements, column_width);
	} else {
		bidy -= right_count - left_offset;
		k = ks[bidy];
		column_width = k + 1;
		if (!isSPD) 
			column_width += k;

		offset = offsets[bidy] + (isSPD ? 0 : k);
		first_row = bidy*partition_size;
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= left_spike_widths[bidy-left_offset]) return;
		idx += g_k - left_spike_widths[bidy-left_offset];
		if (bidy < rest_num) {
			first_row += bidy;
			last_row = first_row + partition_size + 1;
		} else {
			first_row += rest_num;
			last_row = first_row + partition_size;
		}
		fwdElim_offDiag_large_tiled(dA, dB, idx, k, g_k, left_spike_widths[bidy-left_offset], first_row, last_row, offset, a_elements, column_width);
	}
}

template <typename T>
__global__ void
bckElim_spike(int N, int *ks, int g_k, int rightWidth, int *offsets, T *dA, T *dB, int partition_size, int rest_num, int *left_spike_widths, int *right_spike_widths, int *first_rows, bool isSPD, int right_count, int left_count, int left_offset)
{
	__shared__ T a_elements[1024];
	//// __shared__ T a_elements[2];

	int k, offset, first_row, last_row, idx, bidy = blockIdx.y;
	int column_width, factor;

	if (bidy < right_count) {
		k = ks[bidy];
		column_width = k + 1;
		factor = k;
		if (!isSPD) {
			column_width += k;
			factor = 1;
		}

		offset = offsets[bidy] + (isSPD ? 0 : k);
		first_row = bidy*partition_size;
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= right_spike_widths[bidy]) return;
		if (bidy < rest_num) {
			first_row += bidy;
			last_row = first_row + partition_size + 1;
		} else {
			first_row += rest_num;
			last_row = first_row + partition_size;
		}
		offset += column_width * (last_row - 1 - first_row);
        if (first_rows != NULL) {
            first_row = first_rows[bidy];
        }
		bckElim_offDiag_large_tiled(dA, dB, idx, k, g_k, right_spike_widths[bidy], first_row, last_row, offset, a_elements, column_width, factor);
	} else {
		bidy -= right_count - left_offset;
		k = ks[bidy];
		column_width = k + 1;
		factor = k;
		if (!isSPD) {
			column_width += k;
			factor = 1;
		}

		offset = offsets[bidy] + (isSPD ? 0 : k);
		first_row = bidy*partition_size;
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= left_spike_widths[bidy-left_offset]) return;
		idx += g_k - left_spike_widths[bidy-left_offset];
		if (bidy < rest_num) {
			first_row += bidy;
			last_row = first_row + partition_size + 1;
		} else {
			first_row += rest_num;
			last_row = first_row + partition_size;
		}
		offset += column_width * (last_row - 1 - first_row);
		bckElim_offDiag_large_tiled(dA, dB, idx, k, g_k, left_spike_widths[bidy-left_offset], first_row, last_row, offset, a_elements, column_width, factor);
	}

}


// ----------------------------------------------------------------------------
// Forward/backward substitution kernels for calculating the right spikes.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
fwdElim_rightSpike_per_partition(int N, int k, int pivotIdx, T *dA, T *dB, int first_row, int last_row, bool isSPD)
{
	int tid = threadIdx.x, bidx = blockIdx.x * N;
	if (tid >= k) return;
	int it_last = k;
	int column_width = (isSPD ? (k + 1): ((k << 1) + 1));

	for(int i=first_row; i<last_row-k; i++) {
		T tmp = dB[bidx+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i+ttid+1] -= tmp * dA[pivotIdx + ttid + 1];
		pivotIdx += column_width;
		__syncthreads();
	}
	for(int i=(first_row > last_row-k ? first_row : (last_row - k)); i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		it_last --;
		T tmp = dB[bidx+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i+ttid+1] -= tmp * dA[pivotIdx + ttid + 1];
		pivotIdx += column_width;
		__syncthreads();
	}
}

template <typename T>
__global__ void
preBck_rightSpike_divide_per_partition(int N, int k, int pivotIdx, T *dA, T *dB, int first_row, int last_row, bool isSPD)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (first_row + idx >= last_row) return;
	dB[blockIdx.y * N + first_row + idx] /= dA[pivotIdx + idx * (isSPD ? (k + 1): ((k << 1) + 1))];
}

template <typename T>
__global__ void
preBck_offDiag_divide_per_partition(int g_k, int k, int pivotIdx, T *dA, T *dB, int first_row, int last_row)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= g_k) return;
	int row_idx = blockIdx.y + blockIdx.z * gridDim.y;
	if (first_row + row_idx >= last_row) return;
	dB[(row_idx+first_row) * g_k +  idx] /= dA[pivotIdx + row_idx * ((k<<1)+1)];
}

template <typename T>
__global__ void
preBck_offDiag_divide(int N, int g_k, int *ks, int *offsets, T *dA, T *dB, int partSize, int remainder, bool isSPD) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= g_k)
		return;
	int row_idx = blockIdx.y + blockIdx.z * gridDim.y;
	if (row_idx >= N) return;
	
	int partId = row_idx / (partSize + 1), first_row;
	if (partId >= remainder) {
		partId = remainder + (row_idx - remainder * (partSize + 1)) / partSize;
		first_row = partSize * partId + remainder;
	} else {
		first_row = (partSize + 1) * partId;
	}

	int k = ks[partId];
	int pivotIdx = offsets[partId] + (isSPD ?  (k + 1) * (row_idx - first_row) : (k + (2*k+1)*(row_idx - first_row)));

	dB[row_idx * g_k + idx] /= dA[pivotIdx];
}

template <typename T>
__global__ void
bckElim_rightSpike_per_partition(int N, int k, int pivotIdx, T *dA, T *dB, int first_row, int last_row, bool isSPD)
{
	int tid = threadIdx.x, bidx = blockIdx.x * N;
	if (tid >= k) return;

	int it_last = k;
	int column_width = k + 1;
	int factor = k;
	if (!isSPD) {
		column_width += k;
		factor = 1;
	}

	for(int i=last_row-1; i>=first_row + k; i--) {
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i-ttid-1] -= dB[bidx+i] * dA[pivotIdx-(ttid+1) * factor];
		pivotIdx -= column_width;
		__syncthreads();
	}

	for(int i= (k-1+first_row > last_row - 1 ? last_row - 1 : k-1+first_row); i>=first_row; i--) {
		if(tid>=i-first_row) return;
		it_last --;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i-ttid-1] -= dB[bidx+i] * dA[pivotIdx - (ttid + 1) * factor];
		pivotIdx -= column_width;
		__syncthreads();
	}
}

// ----------------------------------------------------------------------------
// Forward/backward substitution kernels for calculating the left spikes.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
fwdElim_leftSpike_per_partition(int N, int k, int bid_delta, int pivotIdx, T *dA, T *dB, int first_row, int last_row, bool isSPD)
{
	int tid = threadIdx.x, bidx = (blockIdx.x+bid_delta) * N;
	if (tid >= k) return;

	int it_last = k;
	int column_width = k + 1;
	if (!isSPD) column_width += k;

	for(int i=first_row; i<last_row-k; i++) {
		T tmp = dB[bidx+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i+ttid+1] -= tmp * dA[pivotIdx + ttid + 1];
		pivotIdx += column_width;
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		if(it_last > last_row-i-1)
			it_last = last_row-i-1;
		T tmp = dB[bidx+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i+ttid+1] -= tmp * dA[pivotIdx + ttid + 1];
		pivotIdx += column_width;
		__syncthreads();
	}
}

template <typename T>
__global__ void
preBck_leftSpike_divide_per_partition(int N, int k, int bid_delta, int pivotIdx, T *dA, T *dB, int first_row, int last_row, bool isSPD)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (first_row + idx >= last_row) return;
	dB[(blockIdx.y+bid_delta) * N + first_row + idx] /= dA[pivotIdx + idx * (isSPD ? (k + 1): ((k<<1)+1))];
}

template <typename T>
__global__ void
bckElim_leftSpike_per_partition(int N, int k, int bid_delta, int pivotIdx, T *dA, T *dB, int first_row, int last_row, bool isSPD)
{
	int tid = threadIdx.x, bidx = (blockIdx.x+bid_delta) * N;
	if (tid >= k) return;

	int it_last = k;
	int column_width = k + 1;
	int factor = k;
	if (!isSPD) {
		column_width += k;
		factor = 1;
	}

	for(int i=last_row-1; i>=k+first_row; i--) {
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i-ttid-1] -= dB[bidx+i] * dA[pivotIdx- (ttid + 1) * factor];
		pivotIdx -= column_width;
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;
		it_last --;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i-ttid-1] -= dB[bidx+i] * dA[pivotIdx - (ttid + 1) * factor];
		pivotIdx -= column_width;
		__syncthreads();
	}
}


} // namespace var
} // namespace device
} // namespace sap


#endif

