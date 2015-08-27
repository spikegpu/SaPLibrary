/** \file sweep_band_var.cuh
 *  Various forward/backward sweep CUDA kernels used for the case of partitions
    with equal bandwidths.
 */

#ifndef SWEEP_BAND_CONST_CUH
#define SWEEP_BAND_CONST_CUH

#include <cuda.h>


namespace sap {
namespace device {


// ----------------------------------------------------------------------------
// CUDA kernels for performing forward elimination sweeps using an exiting
// LU factorization of a full matrix.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
forwardElimLNormal(int N, int k, int partition_size, T *dA, T *dB, int b_partition_size, int b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x;
	int offset = bidx*partition_size*partition_size;

	if(bidx + 1 <= b_rest_num) {
		for(int i=0; i<partition_size-1; i++) {
			if(tid >= partition_size - i - 1) return;
			dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i+tid+1] -= dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i] * dA[i*partition_size + i+tid+1+offset];
			__syncthreads();
		}
	} else {
		for(int i=0; i<partition_size-1; i++) {
			if(tid >= partition_size - i - 1) return;
			dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i+tid+1] -= dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i] * dA[i*partition_size + i+tid+1+offset];
			__syncthreads();
		}
	}
}

template <typename T>
__global__ void
forwardElimLNormal_g512(int N, int k, int partition_size, T *dA, T *dB, int b_partition_size, int b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x;
	int offset = bidx*partition_size*partition_size;

	int it_last = 2*k-1;

	if(bidx + 1 <= b_rest_num) {
		for(int i=0; i<partition_size-1; i++) {
			if(tid >= partition_size - i - 1) return;
			if(it_last > partition_size - i - 1)
				it_last = partition_size - i-1;
			for(int ttid = tid; ttid <  it_last; ttid+=blockDim.x)
				dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i+ttid+1] -= dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i] * dA[i*partition_size + i + ttid+1+offset];
			__syncthreads();
		}
	} else {
		for(int i=0; i<partition_size-1; i++) {
			if(tid >= partition_size - i - 1) return;
			if(it_last > partition_size - i - 1)
				it_last = partition_size - i-1;
			for(int ttid = tid; ttid <  it_last; ttid+=blockDim.x)
				dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i+ttid+1] -= dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i] * dA[i*partition_size + i + ttid+1+offset];
			__syncthreads();
		}
	}
}



// ----------------------------------------------------------------------------
// CUDA kernels for performing backward substitution sweeps using an exiting
// LU factorization of a full matrix.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
backwardElimUNormal(int N, int k, int partition_size, T *dA, T *dB, int b_partition_size, int b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x;
	int offset = bidx*partition_size*partition_size;
	__shared__ T shared_curB;

	if(bidx+1 <= b_rest_num) {
		for(int i=partition_size-1; i>=k; i--) {
			if(tid >= i) return;
			if(tid == 0)
				shared_curB = (dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i] /=dA[partition_size*i+i+offset]);
			__syncthreads();

			dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i-tid-1] -= shared_curB * dA[i*partition_size+i-tid-1+offset];
			__syncthreads();
		}
	} else {
		for(int i=partition_size-1; i>=k; i--) {
			if(tid >= i) return;
			if(tid == 0)
				shared_curB = (dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i] /=dA[partition_size*i+i+offset]);
			__syncthreads();

			dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i-tid-1] -= shared_curB * dA[i*partition_size+i-tid-1+offset];
			__syncthreads();
		}
	}
}

template <typename T>
__global__ void
backwardElimUNormal_g512(int N, int k, int partition_size, T *dA, T *dB, int b_partition_size, int b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x;
	int offset = bidx*partition_size*partition_size;
	__shared__ T shared_curB;
	int it_last = 2*k-1;

	if(bidx + 1 <= b_rest_num) {
		for(int i=partition_size-1; i>=k; i--) {
			if(tid >= i) return;
			if(tid == 0)
				shared_curB = (dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i] /=dA[partition_size*i+i+offset]);
			__syncthreads();

			if(it_last > i)
				it_last = i;
			for(int ttid = tid; ttid <  it_last; ttid+=blockDim.x)
				dB[blockIdx.y*N+(bidx+1)*(b_partition_size+1)-k+i-ttid-1] -= shared_curB * dA[i*partition_size+i-ttid-1+offset];
			__syncthreads();
		}
	} else {
		for(int i=partition_size-1; i>=k; i--) {
			if(tid >= i) return;
			if(tid == 0)
				shared_curB = (dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i] /=dA[partition_size*i+i+offset]);
			__syncthreads();

			if(it_last > i)
				it_last = i;
			for(int ttid = tid; ttid <  it_last; ttid+=blockDim.x)
				dB[blockIdx.y*N+(bidx+1)*b_partition_size+b_rest_num-k+i-ttid-1] -= shared_curB * dA[i*partition_size+i-ttid-1+offset];
			__syncthreads();
		}
	}
}

// ----------------------------------------------------------------------------
// CUDA kernels for performing forward elimination sweeps using an existing
// LU factorization of a banded matrix. These kernels can be used with one or
// more RHS vectors and they can perform either a complete or a partial sweep.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
forwardElimL_general(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
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
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i+ttid+1] -= dB[bidy*N+i] * dA[i*col_width + k + ttid + 1];
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		if(it_last > last_row-i-1)
			it_last = last_row-i-1;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i+ttid+1] -= dB[bidy*N+i] * dA[i*col_width + k + ttid + 1];
		__syncthreads();
	}
}

template <typename T>
__global__ void
fwdElim_sol_forSPD(int N, int k, T *dA, T *dB, int partSize, int remainder)
{
	int col_width = k + 1;
	int first_row = blockIdx.x * partSize;
	int last_row;
	if(blockIdx.x < remainder) {
		first_row += blockIdx.x;
		last_row = first_row + partSize + 1;
	} else {
		first_row += remainder;
		last_row = first_row + partSize;
	}

	int it_last = k;

	for(int i=first_row; i<last_row-k; i++) {
		for(int ttid = threadIdx.x; ttid<it_last; ttid+=blockDim.x)
			dB[i+ttid+1] -= dB[i] * dA[i*col_width + ttid + 1];
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(threadIdx.x >= last_row-i-1) return;
		if(it_last > last_row-i-1)
			it_last = last_row-i-1;
		for(int ttid = threadIdx.x; ttid < it_last; ttid+=blockDim.x)
			dB[i+ttid+1] -= dB[i] * dA[i*col_width + ttid + 1];
		__syncthreads();
	}
}

template <typename T>
__global__ void
fwdElim_sol_medium_forSPD(int N, int k, T *dA, T *dB, int partSize, int remainder)
{
	int col_width = k + 1;
	int first_row = blockIdx.x * partSize;
	int last_row;
	if(blockIdx.x < remainder) {
		first_row += blockIdx.x;
		last_row = first_row + partSize + 1;
	} else {
		first_row += remainder;
		last_row = first_row + partSize;
	}

	int ttid = threadIdx.x;

	for(int i=first_row; i<last_row-k; i++) {
		// if (ttid < ks_col[i])
		dB[i+ttid+1] -= dB[i] * dA[i*col_width + ttid + 1];
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if (ttid >= last_row-i-1) return;
		// if (ttid < ks_col[i])
		dB[i+ttid+1] -= dB[i] * dA[i*col_width + ttid + 1];
		__syncthreads();
	}
}

template <typename T>
__global__ void
forwardElimL_g32(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = (k<<1) + 1;
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
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
		__syncthreads();
	}
}


template <typename T>
__global__ void
forwardElimL(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{

	// The block ID indicates the partition and RHS we are working on.
	// Each thread is responsible for one row (below the main diagonal).
	int iPart = blockIdx.x;
	int iRHS = blockIdx.y;
	int tid = threadIdx.x;
	int stride = 2 * k + 1;

	// Calculate the range of columns in the current partition. The first
	// 'rest_num' partitions have a size of 'partition_size + 1'. All others
	// have a size equal to 'partition_size'.
	int first_col;
	int last_col;

	if (iPart < rest_num) {
		first_col = iPart * partition_size + iPart;
		last_col = first_col + partition_size + 1;
	} else {
		first_col = iPart * partition_size + rest_num;
		last_col = first_col + partition_size;
	}

	// Perform the forward elimination using the L factor from the LU decomposition
	// encoded in dA. Note that on the last 'k' columns, fewer and fewer threads do
	// any work.
	for (int i = first_col; i < last_col - k; i++) {
		dB[iRHS * N + i + tid + 1] -= dB[iRHS * N + i] * dA[i * stride + k + tid + 1];
		__syncthreads();
	}

	for (int i = last_col - k; i < last_col - 1; i++) {
		if (tid >= last_col - i - 1) return;
		dB[iRHS * N + i + tid + 1] -= dB[iRHS * N + i] * dA[i * stride + k + tid + 1];
	}
}

template <typename T>
__global__ void
preBck_sol_divide(int N, int k, T *dA, T *dB, int partition_size, int rest_num, bool saveMem)
{
	int first_row = blockIdx.y*partition_size;
	int last_row;
	if(blockIdx.y < rest_num) {
		first_row += blockIdx.y;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	int column_width = (k << 1) + 1;
	int delta = k;
	if (saveMem) {
		column_width = k + 1;
		delta = 0;
	}
	int pivotIdx = first_row * column_width + delta;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= last_row-first_row)
		return;
	dB[first_row + idx] /= dA[pivotIdx + idx * column_width];
}

template <typename T>
__global__ void
bckElim_sol(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;

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

	for(int i=last_row-1; i>=k+first_row; i--) {
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i-ttid-1] -= dB[bidy*N+i] * dA[i*col_width+k-ttid-1];
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;
		if(it_last > i-first_row)
			it_last = i-first_row;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i-ttid-1] -= dB[bidy*N+i] * dA[i*col_width + k - ttid - 1];
		__syncthreads();
	}
}

template <typename T>
__global__ void
bckElim_sol_forSPD(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x;
	int col_width = k + 1;

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

	for(int i=last_row-1; i>=k+first_row; i--) {
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[i-ttid-1] -= dB[i] * dA[i*col_width-(ttid+1) * k];
		__syncthreads();
	}

	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;
		if(it_last > i-first_row)
			it_last = i-first_row;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[i-ttid-1] -= dB[i] * dA[i*col_width - (ttid + 1) * k];
		__syncthreads();
	}
}

template <typename T>
__global__ void
bckElim_sol_medium_forSPD(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x;
	int col_width = k + 1;

	int first_row = blockIdx.x*partition_size;
	int last_row;
	if(blockIdx.x < rest_num) {
		first_row += blockIdx.x;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}

	for(int i=last_row-1; i>=k+first_row; i--) {
		dB[i-tid-1] -= dB[i] * dA[i*col_width-(tid+1) * k];
		__syncthreads();
	}

	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;
		dB[i-tid-1] -= dB[i] * dA[i*col_width - (tid + 1) * k];
		__syncthreads();
	}
}


template <typename T>
__global__ void
bckElim_sol_medium(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = (k<<1)+ 1;

	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}

	for(int i=last_row-1; i>=k+first_row; i--) {
		dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[i*col_width+k-tid-1];
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;
		dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[i*col_width + k - tid - 1];
		__syncthreads();
	}
}


template <typename T>
__global__ void
bckElim_sol_narrow(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;

	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}

	for(int i=last_row-1; i>=k+first_row; i--) {
		dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[i*col_width+k-tid-1];
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;
		dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[i*col_width + k - tid - 1];
	}
}

// ----------------------------------------------------------------------------
// CUDA kernels for performing backward substitution sweeps using an exiting
// LU factorization of a banded matrix. These kernels can be used with one or
// more RHS vectors and they can perform either a complete or a partial sweep.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
backwardElimU_general(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	__shared__ T shared_curB;
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

	for(int i=last_row-1; i>=k+first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		}
		__syncthreads();
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i-ttid-1] -= shared_curB * dA[i*col_width+k-ttid-1];
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		} 
		__syncthreads();
		if(tid>=i-first_row) return;
		if(it_last > i-first_row)
			it_last = i-first_row;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i-ttid-1] -= shared_curB * dA[i*col_width + k - ttid - 1];
		__syncthreads();
	}
}


template <typename T>
__global__ void
backwardElimU_g32(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = (k<<1)+ 1;
	__shared__ T shared_curB;
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}

	for(int i=last_row-1; i>=k+first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		}
		__syncthreads();

		dB[bidy*N+i-tid-1] -= shared_curB * dA[i*col_width+k-tid-1];
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		} 
		__syncthreads();
		if(tid>=i-first_row) return;
		dB[bidy*N+i-tid-1] -= shared_curB * dA[i*col_width + k - tid - 1];
		__syncthreads();
	}
}


template <typename T>
__global__ void
backwardElimU(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	__shared__ T shared_curB;
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}

	for(int i=last_row-1; i>=k+first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		}
		__syncthreads();
		dB[bidy*N+i-tid-1] -= shared_curB * dA[i*col_width+k-tid-1];
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		} 
		__syncthreads();
		if(tid>=i-first_row) return;
		dB[bidy*N+i-tid-1] -= shared_curB * dA[i*col_width + k - tid - 1];
	}
}




template <typename T>
__global__ void
backwardElimUdWV(int k, T *dA, T *dB, int partition_size, int odd, int divide)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	int first_row = (2*bidx+odd)*(partition_size+1);
	int offset = (2*bidx+odd)*partition_size*partition_size;

	int j=k-1;
	if(divide) {
		__shared__ T shared_curB;
		for(int i=k-1+first_row; i>=first_row; i--, j--) {
			if(tid == 0) {
				shared_curB = (dB[bidy*k+j+offset] /= dA[i*col_width+k]);
			} 
			__syncthreads();
			if(tid>=i-first_row) return;
			dB[bidy*k+j-tid-1+offset] -= shared_curB * dA[i*col_width + k - tid - 1];
		}
	}
	else {
		for(int i=k-1+first_row; i>=first_row; i--, j--) {
			if(tid>=i-first_row) return;
			dB[bidy*k+j-tid-1+offset] -= dB[bidy*k+j+offset] * dA[i*col_width + k - tid - 1];
		}
	}
}

template <typename T>
__global__ void
forwardElimLdWV(int k, T *dA, T *dB, int partition_size, int odd, int divide)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	int last_row = (2*bidx+odd)*(partition_size+1) + partition_size;
	int offset = (2*bidx+odd)*partition_size*partition_size;

	int j=0;
	if(divide) {
		__shared__ T shared_curB;
		for(int i=last_row-k; i<last_row; i++, j++) {
			if(tid == 0)
				shared_curB = (dB[bidy*k+j+offset] /= dA[i*col_width+k]);
			__syncthreads();
			if(tid >= last_row-i-1) return;
			dB[bidy*k+j+tid+1+offset] -= shared_curB * dA[i*col_width + k + tid + 1];
		}
	} else {
		for(int i=last_row-k; i<last_row-1; i++, j++) {
			if(tid >= last_row-i-1) return;
			dB[bidy*k+j+tid+1+offset] -= dB[bidy*k+j+offset] * dA[i*col_width + k + tid + 1];
		}
	}
}

template <typename T>
__global__ void
forwardElimLdWV_g32(int k, T *dA, T *dB, int partition_size, int odd, int divide)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	int last_row = (2*bidx+odd)*(partition_size+1) + partition_size;
	int offset = (2*bidx+odd)*partition_size*partition_size;

	int j=0;
	if(divide) {
		__shared__ T shared_curB;
		for(int i=last_row-k; i<last_row; i++, j++) {
			if(tid == 0)
				shared_curB = (dB[bidy*k+j+offset] /= dA[i*col_width+k]);
			__syncthreads();
			if(tid >= last_row-i-1) return;
			dB[bidy*k+j+tid+1+offset] -= shared_curB * dA[i*col_width + k + tid + 1];
			__syncthreads();
		}
	} else {
		for(int i=last_row-k; i<last_row-1; i++, j++) {
			if(tid >= last_row-i-1) return;
			dB[bidy*k+j+tid+1+offset] -= dB[bidy*k+j+offset] * dA[i*col_width + k + tid + 1];
			__syncthreads();
		}
	}
}

template <typename T>
__global__ void
backwardElimUdWV_g32(int k, T *dA, T *dB, int partition_size, int odd, int divide)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	int first_row = (2*bidx+odd)*(partition_size+1);
	int offset = (2*bidx+odd)*partition_size*partition_size;

	int j=k-1;
	if(divide) {
		__shared__ T shared_curB;
		for(int i=k-1+first_row; i>=first_row; i--, j--) {
			if(tid == 0) {
				shared_curB = (dB[bidy*k+j+offset] /= dA[i*col_width+k]);
			} 
			__syncthreads();
			if(tid>=i-first_row) return;
			dB[bidy*k+j-tid-1+offset] -= shared_curB * dA[i*col_width + k - tid - 1];
			__syncthreads();
		}
	}
	else {
		for(int i=k-1+first_row; i>=first_row; i--, j--) {
			if(tid>=i-first_row) return;
			dB[bidy*k+j-tid-1+offset] -= dB[bidy*k+j+offset] * dA[i*col_width + k - tid - 1];
			__syncthreads();
		}
	}
}

template <typename T>
__global__ void
forwardElimLdWV_general(int k, T *dA, T *dB, int partition_size, int odd, int divide)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	int last_row = (2*blockIdx.x+odd)*(partition_size+1) + partition_size;
	int offset = (2*blockIdx.x+odd)*partition_size*partition_size;

	int j=0;
	if(divide) {
		__shared__ T shared_curB;
		for(int i=last_row-k; i<last_row; i++, j++) {
			if(tid == 0)
				shared_curB = (dB[bidy*k+j+offset] /= dA[i*col_width+k]);
			__syncthreads();
			if(tid >= last_row-i-1) return;
			int it_last = k;
			if(it_last > last_row-i-1)
				it_last = last_row-i-1;
			for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
				dB[bidy*k+j+ttid+1+offset] -= shared_curB * dA[i*col_width + k + ttid + 1];
			__syncthreads();
		}
	} else {
		for(int i=last_row-k; i<last_row-1; i++, j++) {
			if(tid >= last_row-i-1) return;
			int it_last = k;
			if(it_last > last_row-i-1)
				it_last = last_row-i-1;
			for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
				dB[bidy*k+j+ttid+1+offset] -= dB[bidy*k+j+offset] * dA[i*col_width + k + ttid + 1];
			__syncthreads();
		}
	}
}

template <typename T>
__global__ void
backwardElimUdWV_general(int k, T *dA, T *dB, int partition_size, int odd, int divide)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	int first_row = (2*blockIdx.x+odd)*(partition_size+1);
	int offset = (2*blockIdx.x+odd)*partition_size*partition_size;

	int j=k-1;
	if(divide) {
		__shared__ T shared_curB;
		for(int i=k-1+first_row; i>=first_row; i--, j--) {
			if(tid == 0) {
				shared_curB = (dB[bidy*k+j+offset] /= dA[i*col_width+k]);
			} 
			__syncthreads();
			if(tid>=i-first_row) return;
			int step = blockDim.x;
			int it_last = k;
			if(it_last > i-first_row)
				it_last = i-first_row;
			for(int ttid = tid; ttid<it_last; ttid+=step)
				dB[bidy*k+j-ttid-1+offset] -= shared_curB * dA[i*col_width + k - ttid - 1];
			__syncthreads();
		}
	}
	else {
		for(int i=k-1+first_row; i>=first_row; i--, j--) {
			if(tid>=i-first_row) return;
			int step = blockDim.x;
			int it_last = k;
			if(it_last > i-first_row)
				it_last = i-first_row;
			for(int ttid = tid; ttid<it_last; ttid+=step)
				dB[bidy*k+j-ttid-1+offset] -= dB[bidy*k+j+offset] * dA[i*col_width + k - ttid - 1];
			__syncthreads();
		}
	}
}




template <typename T>
__global__ void
forwardElimL_bottom_general(int N, int k, int delta, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	if (delta < last_row-first_row)
		first_row = last_row-delta;

	int it_last = k;

	for(int i=first_row; i<last_row-k; i++) {
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i+ttid+1] -= dB[bidy*N+i] * dA[i*col_width + k + ttid + 1];
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		if(it_last > last_row-i-1)
			it_last = last_row-i-1;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i+ttid+1] -= dB[bidy*N+i] * dA[i*col_width + k + ttid + 1];
		__syncthreads();
	}
}


template <typename T>
__global__ void
backwardElimU_bottom_general(int N, int k, int delta, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	__shared__ T shared_curB;
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	if (delta < last_row-first_row)
		first_row = last_row-delta;
	int it_last = k;

	for(int i=last_row-1; i>=k+first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		}
		__syncthreads();
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i-ttid-1] -= shared_curB * dA[i*col_width+k-ttid-1];
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		} 
		__syncthreads();
		if(tid>=i-first_row) return;
		if(it_last > i-first_row)
			it_last = i-first_row;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidy*N+i-ttid-1] -= shared_curB * dA[i*col_width + k - ttid - 1];
		__syncthreads();
	}
}

template <typename T>
__global__ void
forwardElimL_bottom_g32(int N, int k, int delta, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int col_width = (k<<1) + 1;
	int first_row = blockIdx.x*partition_size;
	int last_row;
	if(blockIdx.x < rest_num) {
		first_row += blockIdx.x;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	if (delta < last_row-first_row)
		first_row = last_row-delta;

	for(int i=first_row; i<last_row-k; i++) {
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
		__syncthreads();
	}
}

template <typename T>
__global__ void
backwardElimU_bottom_g32(int N, int k, int delta, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = (k<<1)+ 1;
	__shared__ T shared_curB;
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	if (delta < last_row-first_row)
		first_row = last_row-delta;

	for(int i=last_row-1; i>=k+first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		}
		__syncthreads();

		dB[bidy*N+i-tid-1] -= shared_curB * dA[i*col_width+k-tid-1];
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		} 
		__syncthreads();
		if(tid>=i-first_row) return;
		dB[bidy*N+i-tid-1] -= shared_curB * dA[i*col_width + k - tid - 1];
		__syncthreads();
	}
}

template <typename T>
__global__ void
forwardElimL_bottom(int N, int k, int delta, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	if (delta < last_row-first_row)
		first_row = last_row-delta;

	for(int i=first_row; i<last_row-k; i++) {
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
	}
}

template <typename T>
__global__ void
backwardElimU_bottom(int N, int k, int delta, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	__shared__ T shared_curB;
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}
	if (delta < last_row-first_row)
		first_row = last_row-delta;

	for(int i=last_row-1; i>=k+first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		}
		__syncthreads();
		dB[bidy*N+i-tid-1] -= shared_curB * dA[i*col_width+k-tid-1];
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid == 0) {
			shared_curB = (dB[bidy*N+i] /= dA[i*col_width+k]);
		} 
		__syncthreads();
		if(tid>=i-first_row) return;
		dB[bidy*N+i-tid-1] -= shared_curB * dA[i*col_width + k - tid - 1];
	}
}

// ----------------------------------------------------------------------------
// CUDA kernels for performing forward elimination sweeps using an existing
// LU/UL factorization of a banded matrix. These kernels can be used with one or
// more RHS vectors and they can perform either a complete or a partial sweep.
// Note that the last partition should be a UL factorization.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
forwardElimL_LU_UL_general(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
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

	if (blockIdx.x < gridDim.x - 1) {
		for(int i=first_row; i<last_row-k; i++) {
			for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
				dB[bidy*N+i+ttid+1] -= dB[bidy*N+i] * dA[i*col_width + k + ttid + 1];
			__syncthreads();
		}
		for(int i=last_row-k; i<last_row-1; i++) {
			if(tid >= last_row-i-1) return;
			if(it_last > last_row-i-1)
				it_last = last_row-i-1;
			for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
				dB[bidy*N+i+ttid+1] -= dB[bidy*N+i] * dA[i*col_width + k + ttid + 1];
			__syncthreads();
		}
	} else {
		for(int i=last_row-1; i>=k+first_row; i--) {
			for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
				dB[bidy*N+i-ttid-1] -= dB[bidy*N+i] * dA[i*col_width+k-ttid-1];
			__syncthreads();
		}
		for(int i=k-1+first_row; i>=first_row; i--) {
			if(tid>=i-first_row) return;
			if(it_last > i-first_row)
				it_last = i-first_row;
			for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
				dB[bidy*N+i-ttid-1] -= dB[bidy*N+i] * dA[i*col_width + k - ttid - 1];
			__syncthreads();
		}
	}
}


template <typename T>
__global__ void
forwardElimL_LU_UL_g32(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = (k<<1) + 1;
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}

	if (bidx < gridDim.x - 1) {
		for(int i=first_row; i<last_row-k; i++) {
			dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
			__syncthreads();
		}
		for(int i=last_row-k; i<last_row-1; i++) {
			if(tid >= last_row-i-1) return;
			dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
			__syncthreads();
		}
	} else {
		for(int i=last_row-1; i>=k+first_row; i--) {
			dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[i*col_width+k-tid-1];
			__syncthreads();
		}
		for(int i=k-1+first_row; i>=first_row; i--) {
			if(tid>=i-first_row) return;
			dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[i*col_width + k - tid - 1];
			__syncthreads();
		}
	}
}


template <typename T>
__global__ void
forwardElimL_LU_UL(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	// The block ID indicates the partition we are working on. Each thread is 
	// responsible for one row (below the main diagonal).
	int iPart = blockIdx.x;
	int tid = threadIdx.x;
	int stride = 2 * k + 1;

	// Calculate the range of columns in the current partition. The first
	// 'rest_num' partitions have a size of 'partition_size + 1'.
	int first_col;
	int last_col;

	if (iPart < rest_num) {
		first_col = iPart * partition_size + iPart;
		last_col = first_col + partition_size + 1;
	} else {
		first_col = iPart * partition_size + rest_num;
		last_col = first_col + partition_size;
	}

	// Perform the sweep.
	if (iPart < gridDim.x - 1) {
		// In all partitions but the last one, this is a forward elimination
		// using the L factor from the LU decomposition encoded in dA.
		// Note that on the last 'k' columns, fewer and fewer threads do work.
		for (int i = first_col; i < last_col - k; i++) {
			dB[i + tid + 1] -= dB[i] * dA[i * stride + k + tid + 1];
			__syncthreads();
		}

		for (int i = last_col - k; i < last_col - 1; i++) {
			if (tid >= last_col - i - 1) return;
			dB[i + tid + 1] -= dB[i] * dA[i * stride + k + tid + 1];
		}

	} else {
		// For the last partition, this is a backward elimination using the
		// U factor from the UL decomposition encoded in dA.
		// Note that on the first 'k' columns, fewer and fewer threads do work.
		for (int i = last_col - 1; i >= k + first_col; i--) {
			dB[i - tid - 1] -= dB[i] * dA[i * stride + k - tid - 1];
			__syncthreads();
		}

		for (int i = k + first_col - 1; i >= first_col; i--) {
			if (tid >= i - first_col) return;
			dB[i - tid - 1] -= dB[i] * dA[i * stride + k - tid - 1];
		}
	}
}


// ----------------------------------------------------------------------------
// CUDA kernels for performing backward substitution sweeps using an exiting
// LU/UL factorization of a banded matrix. These kernels can be used with one or
// more RHS vectors and they can perform either a complete or a partial sweep.
// Note that the last partition should be a UL factorization.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
backwardElimU_LU_UL_general(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
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

	if (blockIdx.x < gridDim.x - 1) {
		for(int i=last_row-1; i>=k+first_row; i--) {
			for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
				dB[bidy*N+i-ttid-1] -= dB[bidy * N + i] * dA[i*col_width+k-ttid-1];
			__syncthreads();
		}
		for(int i=k-1+first_row; i>=first_row; i--) {
			if(tid>=i-first_row) return;
			if(it_last > i-first_row)
				it_last = i-first_row;
			for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
				dB[bidy*N+i-ttid-1] -= dB[bidy * N + i] * dA[i*col_width + k - ttid - 1];
			__syncthreads();
		}
	} else {
		for(int i=first_row; i<last_row-k; i++) {
			for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
				dB[bidy*N+i+ttid+1] -= dB[bidy * N + i] * dA[i*col_width + k + ttid + 1];
			__syncthreads();
		}
		for(int i=last_row-k; i<last_row; i++) {
			if(tid >= last_row-i-1) return;
			if(it_last > last_row-i-1)
				it_last = last_row-i-1;
			for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
				dB[bidy*N+i+ttid+1] -= dB[bidy * N + i] * dA[i*col_width + k + ttid + 1];
			__syncthreads();
		}
	}
}


template <typename T>
__global__ void
backwardElimU_LU_UL_g32(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = (k<<1)+ 1;
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}

	if (blockIdx.x < gridDim.x - 1) {
		for(int i=last_row-1; i>=k+first_row; i--) {
			dB[bidy*N+i-tid-1] -= dB[bidy * N + i] * dA[i*col_width+k-tid-1];
			__syncthreads();
		}
		for(int i=k-1+first_row; i>=first_row; i--) {
			if(tid>=i-first_row) return;
			dB[bidy*N+i-tid-1] -= dB[bidy * N + i] * dA[i*col_width + k - tid - 1];
			__syncthreads();
		}
	} else {
		for(int i=first_row; i<last_row-k; i++) {
			dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
			__syncthreads();
		}
		for(int i=last_row-k; i<last_row; i++) {
			if(tid >= last_row-i-1) return;
			dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
			__syncthreads();
		}
	}
}


template <typename T>
__global__ void
backwardElimU_LU_UL(int N, int k, T *dA, T *dB, int partition_size, int rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
	int col_width = 2*k + 1;
	int first_row = bidx*partition_size;
	int last_row;
	if(bidx < rest_num) {
		first_row += bidx;
		last_row = first_row + partition_size + 1;
	} else {
		first_row += rest_num;
		last_row = first_row + partition_size;
	}

	if (blockIdx.x < gridDim.x - 1) {
		for(int i=last_row-1; i>=k+first_row; i--) {
			dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[i*col_width+k-tid-1];
		}
		for(int i=k-1+first_row; i>=first_row; i--) {
			if(tid>=i-first_row) return;
			dB[bidy*N+i-tid-1] -= dB[bidy*N+i] * dA[i*col_width + k - tid - 1];
		}
	} else {
		for(int i=first_row; i<last_row-k; i++) {
			dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
		}
		for(int i=last_row-k; i<last_row; i++) {
			if(tid >= last_row-i-1) return;
			dB[bidy*N+i+tid+1] -= dB[bidy*N+i] * dA[i*col_width + k + tid + 1];
		}
	}
}



} // namespace device
} // namespace sap


#endif

