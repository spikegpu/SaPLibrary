/** \file factor_band_var.cuh
 *  Various forward/backward sweep CUDA kernels used for the case of partitions
 *  with varying bandwidths.
 */

#ifndef SWEEP_BAND_VAR_CUH
#define SWEEP_BAND_VAR_CUH

#include <cuda.h>


namespace spike {
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
preBck_sol_divide(int N, int *ks, int *offsets, T *dA, T *dB, int partition_size, int rest_num)
{
	int k = ks[blockIdx.y];
	int first_row = blockIdx.y*partition_size;
	int last_row;
	int pivotIdx = offsets[blockIdx.y] + k;
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
	dB[first_row + idx] /= dA[pivotIdx + idx * ((k<<1)+1)];
}

template <typename T>
__device__ void
fwdElim_offDiag_large_tiled(T *dA, T *dB, int idx, int k, int g_k, int r, int first_row, int last_row, int offset, T *a_elements) {

	int step = blockDim.x;
	if ((blockIdx.x+1)*blockDim.x > r)
		step = r - blockIdx.x * blockDim.x;

	int i;
	T tmp0;
	T tmp1;
	T tmp2;
	T tmp3;
	T tmp4;
	T tmp5;
	T tmp6;
	T tmp7;
	T tmp8;
	T tmp9;
	T tmp10;
	T tmp11;
	T tmp12;
	T tmp13;
	T tmp14;
	T tmp15;
	T tmp16;
	T tmp17;
	T tmp18;
	T tmp19;
	T tmp20;
	
	for (i=0; first_row + 20 * (i+1) < last_row; i++) 
	{
		int nextToLoad = g_k*(first_row + 20 * i) + idx;
		tmp0 = dB[nextToLoad]; nextToLoad += g_k;
		tmp1 = dB[nextToLoad]; nextToLoad += g_k;
		tmp2 = dB[nextToLoad]; nextToLoad += g_k;
		tmp3 = dB[nextToLoad]; nextToLoad += g_k;
		tmp4 = dB[nextToLoad]; nextToLoad += g_k;
		tmp5 = dB[nextToLoad]; nextToLoad += g_k;
		tmp6 = dB[nextToLoad]; nextToLoad += g_k;
		tmp7 = dB[nextToLoad]; nextToLoad += g_k;
		tmp8 = dB[nextToLoad]; nextToLoad += g_k;
		tmp9 = dB[nextToLoad]; nextToLoad += g_k;
		tmp10 = dB[nextToLoad]; nextToLoad += g_k;
		tmp11 = dB[nextToLoad]; nextToLoad += g_k;
		tmp12 = dB[nextToLoad]; nextToLoad += g_k;
		tmp13 = dB[nextToLoad]; nextToLoad += g_k;
		tmp14 = dB[nextToLoad]; nextToLoad += g_k;
		tmp15 = dB[nextToLoad]; nextToLoad += g_k;
		tmp16 = dB[nextToLoad]; nextToLoad += g_k;
		tmp17 = dB[nextToLoad]; nextToLoad += g_k;
		tmp18 = dB[nextToLoad]; nextToLoad += g_k;
		tmp19 = dB[nextToLoad]; nextToLoad += g_k;

		int row_to_start = first_row + 20*(i+1)-k-1;
		if (row_to_start < first_row) row_to_start = first_row;

		int sharedOffset = 0;
		for (int j=first_row+20*i-1; j>=row_to_start; j--) {
			int nextToAccess = first_row + 20*i - j - 1;

			if ((nextToAccess & 15) == 0) {
				sharedOffset = 0;
				for (int l = j; l  > j-16 && l>=row_to_start; l--) {
					nextToAccess = first_row + 20*i - l - 1;
					for(int loadIdx = threadIdx.x + nextToAccess; loadIdx < nextToAccess+20 && loadIdx < k; loadIdx += step)
						a_elements[loadIdx+sharedOffset-nextToAccess] = dA[offset + (2*k+1)*(l-first_row) + k + loadIdx + 1];
					sharedOffset += 20;
				}
				sharedOffset = 0;
				__syncthreads();
			}

			tmp20 = dB[g_k * j + idx];
			
			tmp0 -= tmp20 * a_elements[sharedOffset];
			tmp1 -= tmp20 * a_elements[1+sharedOffset];
			tmp2 -= tmp20 * a_elements[2+sharedOffset];
			tmp3 -= tmp20 * a_elements[3+sharedOffset];
			tmp4 -= tmp20 * a_elements[4+sharedOffset];
			tmp5 -= tmp20 * a_elements[5+sharedOffset];
			tmp6 -= tmp20 * a_elements[6+sharedOffset];
			tmp7 -= tmp20 * a_elements[7+sharedOffset];
			tmp8 -= tmp20 * a_elements[8+sharedOffset];
			tmp9 -= tmp20 * a_elements[9+sharedOffset];
			tmp10 -= tmp20 * a_elements[10+sharedOffset];
			tmp11 -= tmp20 * a_elements[11+sharedOffset];
			tmp12 -= tmp20 * a_elements[12+sharedOffset];
			tmp13 -= tmp20 * a_elements[13+sharedOffset];
			tmp14 -= tmp20 * a_elements[14+sharedOffset];
			tmp15 -= tmp20 * a_elements[15+sharedOffset];
			tmp16 -= tmp20 * a_elements[16+sharedOffset];
			tmp17 -= tmp20 * a_elements[17+sharedOffset];
			tmp18 -= tmp20 * a_elements[18+sharedOffset];
			tmp19 -= tmp20 * a_elements[19+sharedOffset];

			sharedOffset += 20;
		}

		sharedOffset = 0;
		for (int j=row_to_start-1; j>row_to_start - 20 && j>=first_row; j--) {
			int nextToAccess = first_row+20*i-j-1;
			for(int loadIdx = threadIdx.x + nextToAccess; loadIdx < nextToAccess+20 && loadIdx < k; loadIdx += step)
				a_elements[loadIdx-nextToAccess+sharedOffset] = dA[offset + k + (2*k+1)*(j-first_row) + loadIdx + 1];
			sharedOffset += 20;
		}
		sharedOffset = 0;
		__syncthreads();

		for (int j=row_to_start-1; j>row_to_start - 20 && j>=first_row; j--) {
			int nextToAccess = first_row+20*i-j-1;

			tmp20 = dB[g_k * j + idx];

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp0 -= tmp20 * a_elements[sharedOffset]; nextToAccess++;
			
			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp1 -= tmp20 * a_elements[sharedOffset+1]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp2 -= tmp20 * a_elements[sharedOffset+2]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp3 -= tmp20 * a_elements[sharedOffset+3]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp4 -= tmp20 * a_elements[sharedOffset+4]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp5 -= tmp20 * a_elements[sharedOffset+5]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp6 -= tmp20 * a_elements[sharedOffset+6]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp7 -= tmp20 * a_elements[sharedOffset+7]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp8 -= tmp20 * a_elements[sharedOffset+8]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp9 -= tmp20 * a_elements[sharedOffset+9]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp10 -= tmp20 * a_elements[sharedOffset+10]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp11 -= tmp20 * a_elements[sharedOffset+11]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp12 -= tmp20 * a_elements[sharedOffset+12]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp13 -= tmp20 * a_elements[sharedOffset+13]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp14 -= tmp20 * a_elements[sharedOffset+14]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp15 -= tmp20 * a_elements[sharedOffset+15]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp16 -= tmp20 * a_elements[sharedOffset+16]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp17 -= tmp20 * a_elements[sharedOffset+17]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp18 -= tmp20 * a_elements[sharedOffset+18]; nextToAccess++;

			if (nextToAccess >= k) {
				sharedOffset += 20;
				continue;
			}
			tmp19 -= tmp20 * a_elements[sharedOffset+19]; nextToAccess++;

			sharedOffset += 20;
		}

		row_to_start = first_row + 20 * i;
		sharedOffset = 0;
		for (int loadCnt = 19; loadCnt > 0; loadCnt--) {
			for (int loadIdx = threadIdx.x; loadIdx < loadCnt; loadIdx += step)
				a_elements[loadIdx+sharedOffset] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
			row_to_start++;
			sharedOffset += loadCnt;
		}
		__syncthreads();

		row_to_start = first_row + 20 * i;
		dB[g_k * row_to_start + idx] = tmp0;
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
		row_to_start ++;

		dB[g_k * row_to_start + idx] = tmp1;
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
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp2;
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
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp3;
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
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp4;
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
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp5;
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
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp6;
		for (int loadIdx = threadIdx.x; loadIdx < 13; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
		__syncthreads();
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
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp7;
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
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp8;
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
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp9;
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
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp10;
		tmp11 -= tmp10 * a_elements[145];
		tmp12 -= tmp10 * a_elements[146];
		tmp13 -= tmp10 * a_elements[147];
		tmp14 -= tmp10 * a_elements[148];
		tmp15 -= tmp10 * a_elements[149];
		tmp16 -= tmp10 * a_elements[150];
		tmp17 -= tmp10 * a_elements[151];
		tmp18 -= tmp10 * a_elements[152];
		tmp19 -= tmp10 * a_elements[153];
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp11;
		tmp12 -= tmp11 * a_elements[154];
		tmp13 -= tmp11 * a_elements[155];
		tmp14 -= tmp11 * a_elements[156];
		tmp15 -= tmp11 * a_elements[157];
		tmp16 -= tmp11 * a_elements[158];
		tmp17 -= tmp11 * a_elements[159];
		tmp18 -= tmp11 * a_elements[160];
		tmp19 -= tmp11 * a_elements[161];
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp12;
		tmp13 -= tmp12 * a_elements[162];
		tmp14 -= tmp12 * a_elements[163];
		tmp15 -= tmp12 * a_elements[164];
		tmp16 -= tmp12 * a_elements[165];
		tmp17 -= tmp12 * a_elements[166];
		tmp18 -= tmp12 * a_elements[167];
		tmp19 -= tmp12 * a_elements[168];
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp13;
		tmp14 -= tmp13 * a_elements[169];
		tmp15 -= tmp13 * a_elements[170];
		tmp16 -= tmp13 * a_elements[171];
		tmp17 -= tmp13 * a_elements[172];
		tmp18 -= tmp13 * a_elements[173];
		tmp19 -= tmp13 * a_elements[174];
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp14;
		tmp15 -= tmp14 * a_elements[175];
		tmp16 -= tmp14 * a_elements[176];
		tmp17 -= tmp14 * a_elements[177];
		tmp18 -= tmp14 * a_elements[178];
		tmp19 -= tmp14 * a_elements[179];
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp15;
		tmp16 -= tmp15 * a_elements[180];
		tmp17 -= tmp15 * a_elements[181];
		tmp18 -= tmp15 * a_elements[182];
		tmp19 -= tmp15 * a_elements[183];
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp16;
		tmp17 -= tmp16 * a_elements[184];
		tmp18 -= tmp16 * a_elements[185];
		tmp19 -= tmp16 * a_elements[186];
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp17;
		tmp18 -= tmp17 * a_elements[187];
		tmp19 -= tmp17 * a_elements[188];
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp18;
		tmp19 -= tmp18 * a_elements[189];
		row_to_start++;

		dB[g_k * row_to_start + idx] = tmp19;
	}

	int row_to_start = first_row + 20*(i+1)-k-1;
	if (row_to_start < first_row) row_to_start = first_row;

	int nextToLoad = g_k*(first_row + 20 * i) + idx;
	if (nextToLoad / g_k < last_row) {
		tmp0 = dB[nextToLoad];
		nextToLoad += g_k;
	}
	if (nextToLoad / g_k < last_row) {
		tmp1 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp2 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp3 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp4 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp5 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp6 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp7 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp8 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp9 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp10 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp11 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp12 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp13 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp14 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp15 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp16 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp17 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp18 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	if (nextToLoad / g_k < last_row) {
		tmp19 = dB[nextToLoad];
		nextToLoad += g_k;
	}

	int sharedOffset = 0;
	for (int j=first_row+20*i-1; j>=row_to_start; j--) {
		int nextToAccess = first_row+20*i-j-1;
		if ((nextToAccess & 15) == 0) {
			sharedOffset = 0;
			for (int l = j; l > j-16 && l>=row_to_start; l--) {
				nextToAccess = first_row+20*i-l-1;
				for(int loadIdx = threadIdx.x + nextToAccess; loadIdx < nextToAccess+20 && loadIdx < k; loadIdx += step)
					a_elements[loadIdx+sharedOffset-nextToAccess] = dA[offset + k + (2*k+1)*(l-first_row) + loadIdx + 1];
				sharedOffset += 20;
			}
			sharedOffset = 0;
			nextToAccess = first_row+20*i;
			__syncthreads();
		}

		nextToAccess = first_row+20*i;
		tmp20 = dB[g_k * j + idx];
		
		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp0 -= tmp20 * a_elements[sharedOffset]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp1 -= tmp20 * a_elements[sharedOffset+1]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp2 -= tmp20 * a_elements[sharedOffset+2]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp3 -= tmp20 * a_elements[sharedOffset+3]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp4 -= tmp20 * a_elements[sharedOffset+4]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp5 -= tmp20 * a_elements[sharedOffset+5]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp6 -= tmp20 * a_elements[sharedOffset+6]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp7 -= tmp20 * a_elements[sharedOffset+7]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp8 -= tmp20 * a_elements[sharedOffset+8]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp9 -= tmp20 * a_elements[sharedOffset+9]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp10 -= tmp20 * a_elements[sharedOffset+10]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp11 -= tmp20 * a_elements[sharedOffset+11]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp12 -= tmp20 * a_elements[sharedOffset+12]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp13 -= tmp20 * a_elements[sharedOffset+13]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp14 -= tmp20 * a_elements[sharedOffset+14]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp15 -= tmp20 * a_elements[sharedOffset+15]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp16 -= tmp20 * a_elements[sharedOffset+16]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp17 -= tmp20 * a_elements[sharedOffset+17]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp18 -= tmp20 * a_elements[sharedOffset+18]; nextToAccess++;

		if (nextToAccess >= last_row)
		{
			sharedOffset += 20;
			continue;
		}
		tmp19 -= tmp20 * a_elements[sharedOffset+19]; nextToAccess++;

		sharedOffset += 20;
	}

	sharedOffset = 0;
	for (int j=row_to_start-1; j>row_to_start - 20 && j>=first_row; j--) {
		int nextToAccess = first_row+20*i-j-1;
		for(int loadIdx = threadIdx.x+nextToAccess; loadIdx < nextToAccess+20 && loadIdx < k; loadIdx += step)
			a_elements[loadIdx-nextToAccess+sharedOffset] = dA[offset + k + (2*k+1)*(j-first_row) + loadIdx + 1];
		sharedOffset += 20;
	}
	sharedOffset = 0;
	__syncthreads();

	for (int j=row_to_start-1; j>row_to_start - 20 && j>=first_row; j--) {
		int nextToAccess = first_row+20*i;

		tmp20 = dB[g_k * j + idx];

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp0 -= tmp20 * a_elements[sharedOffset]; nextToAccess++;
		
		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp1 -= tmp20 * a_elements[sharedOffset+1]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp2 -= tmp20 * a_elements[sharedOffset+2]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp3 -= tmp20 * a_elements[sharedOffset+3]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp4 -= tmp20 * a_elements[sharedOffset+4]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp5 -= tmp20 * a_elements[sharedOffset+5]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp6 -= tmp20 * a_elements[sharedOffset+6]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp7 -= tmp20 * a_elements[sharedOffset+7]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp8 -= tmp20 * a_elements[sharedOffset+8]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp9 -= tmp20 * a_elements[sharedOffset+9]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp10 -= tmp20 * a_elements[sharedOffset+10]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp11 -= tmp20 * a_elements[sharedOffset+11]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp12 -= tmp20 * a_elements[sharedOffset+12]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp13 -= tmp20 * a_elements[sharedOffset+13]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp14 -= tmp20 * a_elements[sharedOffset+14]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp15 -= tmp20 * a_elements[sharedOffset+15]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp16 -= tmp20 * a_elements[sharedOffset+16]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp17 -= tmp20 * a_elements[sharedOffset+17]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp18 -= tmp20 * a_elements[sharedOffset+18]; nextToAccess++;

		if (nextToAccess - j - 1 >= k || nextToAccess >= last_row) {
			sharedOffset += 20;
			continue;
		}
		tmp19 -= tmp20 * a_elements[sharedOffset+19]; nextToAccess++;

		sharedOffset += 20;
	}

	row_to_start = first_row + 20 * i;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp0;
	for (int loadIdx = threadIdx.x; loadIdx < 19; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp1 -= tmp0 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp2 -= tmp0 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp3 -= tmp0 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp4 -= tmp0 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp5 -= tmp0 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp6 -= tmp0 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp7 -= tmp0 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp8 -= tmp0 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp9 -= tmp0 * a_elements[8];
	if (row_to_start + 10 < last_row)
		tmp10 -= tmp0 * a_elements[9];
	if (row_to_start + 11 < last_row)
		tmp11 -= tmp0 * a_elements[10];
	if (row_to_start + 12 < last_row)
		tmp12 -= tmp0 * a_elements[11];
	if (row_to_start + 13 < last_row)
		tmp13 -= tmp0 * a_elements[12];
	if (row_to_start + 14 < last_row)
		tmp14 -= tmp0 * a_elements[13];
	if (row_to_start + 15 < last_row)
		tmp15 -= tmp0 * a_elements[14];
	if (row_to_start + 16 < last_row)
		tmp16 -= tmp0 * a_elements[15];
	if (row_to_start + 17 < last_row)
		tmp17 -= tmp0 * a_elements[16];
	if (row_to_start + 18 < last_row)
		tmp18 -= tmp0 * a_elements[17];
	if (row_to_start + 19 < last_row)
		tmp19 -= tmp0 * a_elements[18];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp1;
	for (int loadIdx = threadIdx.x; loadIdx < 18; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp2 -= tmp1 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp3 -= tmp1 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp4 -= tmp1 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp5 -= tmp1 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp6 -= tmp1 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp7 -= tmp1 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp8 -= tmp1 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp9 -= tmp1 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp10 -= tmp1 * a_elements[8];
	if (row_to_start + 10 < last_row)
		tmp11 -= tmp1 * a_elements[9];
	if (row_to_start + 11 < last_row)
		tmp12 -= tmp1 * a_elements[10];
	if (row_to_start + 12 < last_row)
		tmp13 -= tmp1 * a_elements[11];
	if (row_to_start + 13 < last_row)
		tmp14 -= tmp1 * a_elements[12];
	if (row_to_start + 14 < last_row)
		tmp15 -= tmp1 * a_elements[13];
	if (row_to_start + 15 < last_row)
		tmp16 -= tmp1 * a_elements[14];
	if (row_to_start + 16 < last_row)
		tmp17 -= tmp1 * a_elements[15];
	if (row_to_start + 17 < last_row)
		tmp18 -= tmp1 * a_elements[16];
	if (row_to_start + 18 < last_row)
		tmp19 -= tmp1 * a_elements[17];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp2;
	for (int loadIdx = threadIdx.x; loadIdx < 17; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp3 -= tmp2 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp4 -= tmp2 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp5 -= tmp2 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp6 -= tmp2 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp7 -= tmp2 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp8 -= tmp2 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp9 -= tmp2 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp10 -= tmp2 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp11 -= tmp2 * a_elements[8];
	if (row_to_start + 10 < last_row)
		tmp12 -= tmp2 * a_elements[9];
	if (row_to_start + 11 < last_row)
		tmp13 -= tmp2 * a_elements[10];
	if (row_to_start + 12 < last_row)
		tmp14 -= tmp2 * a_elements[11];
	if (row_to_start + 13 < last_row)
		tmp15 -= tmp2 * a_elements[12];
	if (row_to_start + 14 < last_row)
		tmp16 -= tmp2 * a_elements[13];
	if (row_to_start + 15 < last_row)
		tmp17 -= tmp2 * a_elements[14];
	if (row_to_start + 16 < last_row)
		tmp18 -= tmp2 * a_elements[15];
	if (row_to_start + 17 < last_row)
		tmp19 -= tmp2 * a_elements[16];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp3;
	for (int loadIdx = threadIdx.x; loadIdx < 16; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp4 -= tmp3 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp5 -= tmp3 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp6 -= tmp3 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp7 -= tmp3 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp8 -= tmp3 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp9 -= tmp3 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp10 -= tmp3 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp11 -= tmp3 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp12 -= tmp3 * a_elements[8];
	if (row_to_start + 10 < last_row)
		tmp13 -= tmp3 * a_elements[9];
	if (row_to_start + 11 < last_row)
		tmp14 -= tmp3 * a_elements[10];
	if (row_to_start + 12 < last_row)
		tmp15 -= tmp3 * a_elements[11];
	if (row_to_start + 13 < last_row)
		tmp16 -= tmp3 * a_elements[12];
	if (row_to_start + 14 < last_row)
		tmp17 -= tmp3 * a_elements[13];
	if (row_to_start + 15 < last_row)
		tmp18 -= tmp3 * a_elements[14];
	if (row_to_start + 16 < last_row)
		tmp19 -= tmp3 * a_elements[15];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp4;
	for (int loadIdx = threadIdx.x; loadIdx < 15; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp5 -= tmp4 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp6 -= tmp4 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp7 -= tmp4 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp8 -= tmp4 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp9 -= tmp4 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp10 -= tmp4 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp11 -= tmp4 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp12 -= tmp4 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp13 -= tmp4 * a_elements[8];
	if (row_to_start + 10 < last_row)
		tmp14 -= tmp4 * a_elements[9];
	if (row_to_start + 11 < last_row)
		tmp15 -= tmp4 * a_elements[10];
	if (row_to_start + 12 < last_row)
		tmp16 -= tmp4 * a_elements[11];
	if (row_to_start + 13 < last_row)
		tmp17 -= tmp4 * a_elements[12];
	if (row_to_start + 14 < last_row)
		tmp18 -= tmp4 * a_elements[13];
	if (row_to_start + 15 < last_row)
		tmp19 -= tmp4 * a_elements[14];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp5;
	for (int loadIdx = threadIdx.x; loadIdx < 14; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp6 -= tmp5 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp7 -= tmp5 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp8 -= tmp5 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp9 -= tmp5 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp10 -= tmp5 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp11 -= tmp5 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp12 -= tmp5 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp13 -= tmp5 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp14 -= tmp5 * a_elements[8];
	if (row_to_start + 10 < last_row)
		tmp15 -= tmp5 * a_elements[9];
	if (row_to_start + 11 < last_row)
		tmp16 -= tmp5 * a_elements[10];
	if (row_to_start + 12 < last_row)
		tmp17 -= tmp5 * a_elements[11];
	if (row_to_start + 13 < last_row)
		tmp18 -= tmp5 * a_elements[12];
	if (row_to_start + 14 < last_row)
		tmp19 -= tmp5 * a_elements[13];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp6;
	for (int loadIdx = threadIdx.x; loadIdx < 13; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp7 -= tmp6 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp8 -= tmp6 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp9 -= tmp6 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp10 -= tmp6 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp11 -= tmp6 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp12 -= tmp6 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp13 -= tmp6 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp14 -= tmp6 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp15 -= tmp6 * a_elements[8];
	if (row_to_start + 10 < last_row)
		tmp16 -= tmp6 * a_elements[9];
	if (row_to_start + 11 < last_row)
		tmp17 -= tmp6 * a_elements[10];
	if (row_to_start + 12 < last_row)
		tmp18 -= tmp6 * a_elements[11];
	if (row_to_start + 13 < last_row)
		tmp19 -= tmp6 * a_elements[12];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp7;
	for (int loadIdx = threadIdx.x; loadIdx < 12; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp8 -= tmp7 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp9 -= tmp7 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp10 -= tmp7 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp11 -= tmp7 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp12 -= tmp7 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp13 -= tmp7 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp14 -= tmp7 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp15 -= tmp7 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp16 -= tmp7 * a_elements[8];
	if (row_to_start + 10 < last_row)
		tmp17 -= tmp7 * a_elements[9];
	if (row_to_start + 11 < last_row)
		tmp18 -= tmp7 * a_elements[10];
	if (row_to_start + 12 < last_row)
		tmp19 -= tmp7 * a_elements[11];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp8;
	for (int loadIdx = threadIdx.x; loadIdx < 11; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp9 -= tmp8 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp10 -= tmp8 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp11 -= tmp8 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp12 -= tmp8 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp13 -= tmp8 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp14 -= tmp8 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp15 -= tmp8 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp16 -= tmp8 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp17 -= tmp8 * a_elements[8];
	if (row_to_start + 10 < last_row)
		tmp18 -= tmp8 * a_elements[9];
	if (row_to_start + 11 < last_row)
		tmp19 -= tmp8 * a_elements[10];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp9;
	for (int loadIdx = threadIdx.x; loadIdx < 10; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp10 -= tmp9 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp11 -= tmp9 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp12 -= tmp9 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp13 -= tmp9 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp14 -= tmp9 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp15 -= tmp9 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp16 -= tmp9 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp17 -= tmp9 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp18 -= tmp9 * a_elements[8];
	if (row_to_start + 10 < last_row)
		tmp19 -= tmp9 * a_elements[9];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp10;
	for (int loadIdx = threadIdx.x; loadIdx < 9; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp11 -= tmp10 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp12 -= tmp10 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp13 -= tmp10 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp14 -= tmp10 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp15 -= tmp10 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp16 -= tmp10 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp17 -= tmp10 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp18 -= tmp10 * a_elements[7];
	if (row_to_start + 9 < last_row)
		tmp19 -= tmp10 * a_elements[8];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp11;
	for (int loadIdx = threadIdx.x; loadIdx < 8; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp12 -= tmp11 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp13 -= tmp11 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp14 -= tmp11 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp15 -= tmp11 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp16 -= tmp11 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp17 -= tmp11 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp18 -= tmp11 * a_elements[6];
	if (row_to_start + 8 < last_row)
		tmp19 -= tmp11 * a_elements[7];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp12;
	for (int loadIdx = threadIdx.x; loadIdx < 7; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp13 -= tmp12 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp14 -= tmp12 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp15 -= tmp12 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp16 -= tmp12 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp17 -= tmp12 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp18 -= tmp12 * a_elements[5];
	if (row_to_start + 7 < last_row)
		tmp19 -= tmp12 * a_elements[6];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp13;
	for (int loadIdx = threadIdx.x; loadIdx < 6; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp14 -= tmp13 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp15 -= tmp13 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp16 -= tmp13 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp17 -= tmp13 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp18 -= tmp13 * a_elements[4];
	if (row_to_start + 6 < last_row)
		tmp19 -= tmp13 * a_elements[5];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp14;
	for (int loadIdx = threadIdx.x; loadIdx < 5; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp15 -= tmp14 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp16 -= tmp14 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp17 -= tmp14 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp18 -= tmp14 * a_elements[3];
	if (row_to_start + 5 < last_row)
		tmp19 -= tmp14 * a_elements[4];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp15;
	for (int loadIdx = threadIdx.x; loadIdx < 4; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp16 -= tmp15 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp17 -= tmp15 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp18 -= tmp15 * a_elements[2];
	if (row_to_start + 4 < last_row)
		tmp19 -= tmp15 * a_elements[3];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp16;
	for (int loadIdx = threadIdx.x; loadIdx < 3; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp17 -= tmp16 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp18 -= tmp16 * a_elements[1];
	if (row_to_start + 3 < last_row)
		tmp19 -= tmp16 * a_elements[2];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp17;
	for (int loadIdx = threadIdx.x; loadIdx < 2; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp18 -= tmp17 * a_elements[0];
	if (row_to_start + 2 < last_row)
		tmp19 -= tmp17 * a_elements[1];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp18;
	for (int loadIdx = threadIdx.x; loadIdx < 1; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k + (2*k+1)*(row_to_start - first_row) + loadIdx + 1];
	__syncthreads();
	if (row_to_start + 1 < last_row)
		tmp19 -= tmp18 * a_elements[0];
	row_to_start++;

	if (row_to_start >= last_row) return;
	dB[g_k * row_to_start + idx] = tmp19;
}

template <typename T>
__device__ void
bckElim_offDiag_large_tiled(T *dA, T *dB, int idx, int k, int g_k, int r, int first_row, int last_row, int offset, T *a_elements) {
	int step = blockDim.x;
	if ((blockIdx.x+1)*blockDim.x > r)
		step = r - blockIdx.x * blockDim.x;

	int i;
	T tmp0;
	T tmp1;
	T tmp2;
	T tmp3;
	T tmp4;
	T tmp5;
	T tmp6;
	T tmp7;
	T tmp8;
	T tmp9;
	T tmp10;
	T tmp11;
	T tmp12;
	T tmp13;
	T tmp14;
	T tmp15;
	T tmp16;
	T tmp17;
	T tmp18;
	T tmp19;
	T tmp20;
	
	for (i=0; last_row - 1 -  20 * (i+1) >= first_row; i++) 
	{
		int nextToLoad = g_k*(last_row - 1 - 20 * i) + idx;
		tmp0 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp1 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp2 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp3 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp4 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp5 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp6 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp7 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp8 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp9 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp10 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp11 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp12 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp13 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp14 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp15 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp16 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp17 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp18 = dB[nextToLoad]; nextToLoad -= g_k;
		tmp19 = dB[nextToLoad]; nextToLoad -= g_k;

		int row_to_start = last_row - 20*(i+1) + k;
		if (row_to_start >= last_row) row_to_start = last_row - 1;

		int sharedOffset = 0;

		for (int j=last_row-20*i; j<=row_to_start; j++) {
			int nextToAccess = j- (last_row-20*i);
			if ((nextToAccess & 15) == 0) {
				sharedOffset = 0;
				for (int l = j; l < j+16 && l <= row_to_start; l++) {
					nextToAccess = l-(last_row - 20*i);
					for(int loadIdx = threadIdx.x + nextToAccess; loadIdx < nextToAccess+20 && loadIdx < k; loadIdx += step)
						a_elements[loadIdx-nextToAccess+sharedOffset] = dA[offset - (2*k+1)*(last_row - 1 - l) + k - loadIdx - 1];
					sharedOffset += 20;
				}
				sharedOffset = 0;
				__syncthreads();
			}

			tmp20 = dB[g_k * j + idx];
			
			tmp0 -= tmp20 * a_elements[sharedOffset];
			tmp1 -= tmp20 * a_elements[sharedOffset+1];
			tmp2 -= tmp20 * a_elements[sharedOffset+2];
			tmp3 -= tmp20 * a_elements[sharedOffset+3];
			tmp4 -= tmp20 * a_elements[sharedOffset+4];
			tmp5 -= tmp20 * a_elements[sharedOffset+5];
			tmp6 -= tmp20 * a_elements[sharedOffset+6];
			tmp7 -= tmp20 * a_elements[sharedOffset+7];
			tmp8 -= tmp20 * a_elements[sharedOffset+8];
			tmp9 -= tmp20 * a_elements[sharedOffset+9];
			tmp10 -= tmp20 * a_elements[sharedOffset+10];
			tmp11 -= tmp20 * a_elements[sharedOffset+11];
			tmp12 -= tmp20 * a_elements[sharedOffset+12];
			tmp13 -= tmp20 * a_elements[sharedOffset+13];
			tmp14 -= tmp20 * a_elements[sharedOffset+14];
			tmp15 -= tmp20 * a_elements[sharedOffset+15];
			tmp16 -= tmp20 * a_elements[sharedOffset+16];
			tmp17 -= tmp20 * a_elements[sharedOffset+17];
			tmp18 -= tmp20 * a_elements[sharedOffset+18];
			tmp19 -= tmp20 * a_elements[sharedOffset+19];

			sharedOffset += 20;
		}

		sharedOffset = 0;
		for (int j=row_to_start+1; j<row_to_start + 20 && j < last_row; j++) {
			int nextToAccess = j - (last_row - 20*i);
			for(int loadIdx = threadIdx.x + nextToAccess; loadIdx < nextToAccess+20 && loadIdx < k; loadIdx += step)
				a_elements[loadIdx-nextToAccess+sharedOffset] = dA[offset + k - (2*k+1)*(last_row - 1 - j) - loadIdx - 1];
			sharedOffset += 20;
		}
		sharedOffset = 0;
		__syncthreads();

		for (int j=row_to_start+1; j<row_to_start + 20 && j < last_row; j++, sharedOffset+=20) {

			tmp20 = dB[g_k * j + idx];
			int nextToAccess = j - (last_row - 20*i);

			if (nextToAccess >= k)
				continue;
			tmp0 -= tmp20 * a_elements[sharedOffset]; nextToAccess++;
			
			if (nextToAccess >= k)
				continue;
			tmp1 -= tmp20 * a_elements[sharedOffset+1]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp2 -= tmp20 * a_elements[sharedOffset+2]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp3 -= tmp20 * a_elements[sharedOffset+3]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp4 -= tmp20 * a_elements[sharedOffset+4]; nextToAccess++;
			
			if (nextToAccess >= k)
				continue;
			tmp5 -= tmp20 * a_elements[sharedOffset+5]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp6 -= tmp20 * a_elements[sharedOffset+6]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp7 -= tmp20 * a_elements[sharedOffset+7]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp8 -= tmp20 * a_elements[sharedOffset+8]; nextToAccess++;
			
			if (nextToAccess >= k)
				continue;
			tmp9 -= tmp20 * a_elements[sharedOffset+9]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp10 -= tmp20 * a_elements[sharedOffset+10]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp11 -= tmp20 * a_elements[sharedOffset+11]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp12 -= tmp20 * a_elements[sharedOffset+12]; nextToAccess++;
			
			if (nextToAccess >= k)
				continue;
			tmp13 -= tmp20 * a_elements[sharedOffset+13]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp14 -= tmp20 * a_elements[sharedOffset+14]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp15 -= tmp20 * a_elements[sharedOffset+15]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp16 -= tmp20 * a_elements[sharedOffset+16]; nextToAccess++;
			
			if (nextToAccess >= k)
				continue;
			tmp17 -= tmp20 * a_elements[sharedOffset+17]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp18 -= tmp20 * a_elements[sharedOffset+18]; nextToAccess++;

			if (nextToAccess >= k)
				continue;
			tmp19 -= tmp20 * a_elements[sharedOffset+19]; nextToAccess++;
		}

		row_to_start = last_row - 1 - 20 * i;
		for (int loadIdx = threadIdx.x; loadIdx < 19; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		dB[g_k * row_to_start + idx] = tmp0;
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
		row_to_start --;

		dB[g_k * row_to_start + idx] = tmp1;
		for (int loadIdx = threadIdx.x; loadIdx < 18; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp2 -= tmp1 * a_elements[0];
		tmp3 -= tmp1 * a_elements[1];
		tmp4 -= tmp1 * a_elements[2];
		tmp5 -= tmp1 * a_elements[3];
		tmp6 -= tmp1 * a_elements[4];
		tmp7 -= tmp1 * a_elements[5];
		tmp8 -= tmp1 * a_elements[6];
		tmp9 -= tmp1 * a_elements[7];
		tmp10 -= tmp1 * a_elements[8];
		tmp11 -= tmp1 * a_elements[9];
		tmp12 -= tmp1 * a_elements[10];
		tmp13 -= tmp1 * a_elements[11];
		tmp14 -= tmp1 * a_elements[12];
		tmp15 -= tmp1 * a_elements[13];
		tmp16 -= tmp1 * a_elements[14];
		tmp17 -= tmp1 * a_elements[15];
		tmp18 -= tmp1 * a_elements[16];
		tmp19 -= tmp1 * a_elements[17];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp2;
		for (int loadIdx = threadIdx.x; loadIdx < 17; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp3 -= tmp2 * a_elements[0];
		tmp4 -= tmp2 * a_elements[1];
		tmp5 -= tmp2 * a_elements[2];
		tmp6 -= tmp2 * a_elements[3];
		tmp7 -= tmp2 * a_elements[4];
		tmp8 -= tmp2 * a_elements[5];
		tmp9 -= tmp2 * a_elements[6];
		tmp10 -= tmp2 * a_elements[7];
		tmp11 -= tmp2 * a_elements[8];
		tmp12 -= tmp2 * a_elements[9];
		tmp13 -= tmp2 * a_elements[10];
		tmp14 -= tmp2 * a_elements[11];
		tmp15 -= tmp2 * a_elements[12];
		tmp16 -= tmp2 * a_elements[13];
		tmp17 -= tmp2 * a_elements[14];
		tmp18 -= tmp2 * a_elements[15];
		tmp19 -= tmp2 * a_elements[16];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp3;
		for (int loadIdx = threadIdx.x; loadIdx < 16; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp4 -= tmp3 * a_elements[0];
		tmp5 -= tmp3 * a_elements[1];
		tmp6 -= tmp3 * a_elements[2];
		tmp7 -= tmp3 * a_elements[3];
		tmp8 -= tmp3 * a_elements[4];
		tmp9 -= tmp3 * a_elements[5];
		tmp10 -= tmp3 * a_elements[6];
		tmp11 -= tmp3 * a_elements[7];
		tmp12 -= tmp3 * a_elements[8];
		tmp13 -= tmp3 * a_elements[9];
		tmp14 -= tmp3 * a_elements[10];
		tmp15 -= tmp3 * a_elements[11];
		tmp16 -= tmp3 * a_elements[12];
		tmp17 -= tmp3 * a_elements[13];
		tmp18 -= tmp3 * a_elements[14];
		tmp19 -= tmp3 * a_elements[15];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp4;
		for (int loadIdx = threadIdx.x; loadIdx < 15; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp5 -= tmp4 * a_elements[0];
		tmp6 -= tmp4 * a_elements[1];
		tmp7 -= tmp4 * a_elements[2];
		tmp8 -= tmp4 * a_elements[3];
		tmp9 -= tmp4 * a_elements[4];
		tmp10 -= tmp4 * a_elements[5];
		tmp11 -= tmp4 * a_elements[6];
		tmp12 -= tmp4 * a_elements[7];
		tmp13 -= tmp4 * a_elements[8];
		tmp14 -= tmp4 * a_elements[9];
		tmp15 -= tmp4 * a_elements[10];
		tmp16 -= tmp4 * a_elements[11];
		tmp17 -= tmp4 * a_elements[12];
		tmp18 -= tmp4 * a_elements[13];
		tmp19 -= tmp4 * a_elements[14];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp5;
		for (int loadIdx = threadIdx.x; loadIdx < 14; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp6 -= tmp5 * a_elements[0];
		tmp7 -= tmp5 * a_elements[1];
		tmp8 -= tmp5 * a_elements[2];
		tmp9 -= tmp5 * a_elements[3];
		tmp10 -= tmp5 * a_elements[4];
		tmp11 -= tmp5 * a_elements[5];
		tmp12 -= tmp5 * a_elements[6];
		tmp13 -= tmp5 * a_elements[7];
		tmp14 -= tmp5 * a_elements[8];
		tmp15 -= tmp5 * a_elements[9];
		tmp16 -= tmp5 * a_elements[10];
		tmp17 -= tmp5 * a_elements[11];
		tmp18 -= tmp5 * a_elements[12];
		tmp19 -= tmp5 * a_elements[13];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp6;
		for (int loadIdx = threadIdx.x; loadIdx < 13; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp7 -= tmp6 * a_elements[0];
		tmp8 -= tmp6 * a_elements[1];
		tmp9 -= tmp6 * a_elements[2];
		tmp10 -= tmp6 * a_elements[3];
		tmp11 -= tmp6 * a_elements[4];
		tmp12 -= tmp6 * a_elements[5];
		tmp13 -= tmp6 * a_elements[6];
		tmp14 -= tmp6 * a_elements[7];
		tmp15 -= tmp6 * a_elements[8];
		tmp16 -= tmp6 * a_elements[9];
		tmp17 -= tmp6 * a_elements[10];
		tmp18 -= tmp6 * a_elements[11];
		tmp19 -= tmp6 * a_elements[12];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp7;
		for (int loadIdx = threadIdx.x; loadIdx < 12; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp8 -= tmp7 * a_elements[0];
		tmp9 -= tmp7 * a_elements[1];
		tmp10 -= tmp7 * a_elements[2];
		tmp11 -= tmp7 * a_elements[3];
		tmp12 -= tmp7 * a_elements[4];
		tmp13 -= tmp7 * a_elements[5];
		tmp14 -= tmp7 * a_elements[6];
		tmp15 -= tmp7 * a_elements[7];
		tmp16 -= tmp7 * a_elements[8];
		tmp17 -= tmp7 * a_elements[9];
		tmp18 -= tmp7 * a_elements[10];
		tmp19 -= tmp7 * a_elements[11];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp8;
		for (int loadIdx = threadIdx.x; loadIdx < 11; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp9 -= tmp8 * a_elements[0];
		tmp10 -= tmp8 * a_elements[1];
		tmp11 -= tmp8 * a_elements[2];
		tmp12 -= tmp8 * a_elements[3];
		tmp13 -= tmp8 * a_elements[4];
		tmp14 -= tmp8 * a_elements[5];
		tmp15 -= tmp8 * a_elements[6];
		tmp16 -= tmp8 * a_elements[7];
		tmp17 -= tmp8 * a_elements[8];
		tmp18 -= tmp8 * a_elements[9];
		tmp19 -= tmp8 * a_elements[10];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp9;
		for (int loadIdx = threadIdx.x; loadIdx < 10; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp10 -= tmp9 * a_elements[0];
		tmp11 -= tmp9 * a_elements[1];
		tmp12 -= tmp9 * a_elements[2];
		tmp13 -= tmp9 * a_elements[3];
		tmp14 -= tmp9 * a_elements[4];
		tmp15 -= tmp9 * a_elements[5];
		tmp16 -= tmp9 * a_elements[6];
		tmp17 -= tmp9 * a_elements[7];
		tmp18 -= tmp9 * a_elements[8];
		tmp19 -= tmp9 * a_elements[9];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp10;
		for (int loadIdx = threadIdx.x; loadIdx < 9; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp11 -= tmp10 * a_elements[0];
		tmp12 -= tmp10 * a_elements[1];
		tmp13 -= tmp10 * a_elements[2];
		tmp14 -= tmp10 * a_elements[3];
		tmp15 -= tmp10 * a_elements[4];
		tmp16 -= tmp10 * a_elements[5];
		tmp17 -= tmp10 * a_elements[6];
		tmp18 -= tmp10 * a_elements[7];
		tmp19 -= tmp10 * a_elements[8];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp11;
		for (int loadIdx = threadIdx.x; loadIdx < 8; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp12 -= tmp11 * a_elements[0];
		tmp13 -= tmp11 * a_elements[1];
		tmp14 -= tmp11 * a_elements[2];
		tmp15 -= tmp11 * a_elements[3];
		tmp16 -= tmp11 * a_elements[4];
		tmp17 -= tmp11 * a_elements[5];
		tmp18 -= tmp11 * a_elements[6];
		tmp19 -= tmp11 * a_elements[7];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp12;
		for (int loadIdx = threadIdx.x; loadIdx < 7; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp13 -= tmp12 * a_elements[0];
		tmp14 -= tmp12 * a_elements[1];
		tmp15 -= tmp12 * a_elements[2];
		tmp16 -= tmp12 * a_elements[3];
		tmp17 -= tmp12 * a_elements[4];
		tmp18 -= tmp12 * a_elements[5];
		tmp19 -= tmp12 * a_elements[6];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp13;
		for (int loadIdx = threadIdx.x; loadIdx < 6; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp14 -= tmp13 * a_elements[0];
		tmp15 -= tmp13 * a_elements[1];
		tmp16 -= tmp13 * a_elements[2];
		tmp17 -= tmp13 * a_elements[3];
		tmp18 -= tmp13 * a_elements[4];
		tmp19 -= tmp13 * a_elements[5];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp14;
		for (int loadIdx = threadIdx.x; loadIdx < 5; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp15 -= tmp14 * a_elements[0];
		tmp16 -= tmp14 * a_elements[1];
		tmp17 -= tmp14 * a_elements[2];
		tmp18 -= tmp14 * a_elements[3];
		tmp19 -= tmp14 * a_elements[4];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp15;
		for (int loadIdx = threadIdx.x; loadIdx < 4; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp16 -= tmp15 * a_elements[0];
		tmp17 -= tmp15 * a_elements[1];
		tmp18 -= tmp15 * a_elements[2];
		tmp19 -= tmp15 * a_elements[3];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp16;
		for (int loadIdx = threadIdx.x; loadIdx < 3; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp17 -= tmp16 * a_elements[0];
		tmp18 -= tmp16 * a_elements[1];
		tmp19 -= tmp16 * a_elements[2];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp17;
		for (int loadIdx = threadIdx.x; loadIdx < 2; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp18 -= tmp17 * a_elements[0];
		tmp19 -= tmp17 * a_elements[1];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp18;
		for (int loadIdx = threadIdx.x; loadIdx < 1; loadIdx += step)
			a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
		__syncthreads();
		tmp19 -= tmp18 * a_elements[0];
		row_to_start--;

		dB[g_k * row_to_start + idx] = tmp19;
	}

	int row_to_start = last_row - 20*(i+1)+k;
	if (row_to_start >= last_row) row_to_start = last_row - 1;

	int nextToLoad = g_k*(last_row - 1 - 20 * i) + idx;
	if (nextToLoad / g_k >= first_row) {
		tmp0 = dB[nextToLoad];
		nextToLoad -= g_k;
	}
	if (nextToLoad / g_k >= first_row) {
		tmp1 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp2 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp3 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp4 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp5 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp6 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp7 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp8 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp9 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp10 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp11 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp12 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp13 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp14 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp15 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp16 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp17 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp18 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	if (nextToLoad / g_k >= first_row) {
		tmp19 = dB[nextToLoad];
		nextToLoad -= g_k;
	}

	int sharedOffset = 0;
	for (int j=last_row-20*i; j<=row_to_start; j++, sharedOffset += 20) {
		int nextToAccess = last_row-1-20*i;

		if (((j-nextToAccess-1) & 15) == 0) {
			sharedOffset = 0;
			for (int l = j; l < j+16 && l <=row_to_start; l++) {
				for(int loadIdx = threadIdx.x + l-nextToAccess-1; loadIdx < l-nextToAccess+19 && loadIdx < k; loadIdx += step)
					a_elements[loadIdx-l+nextToAccess+1+sharedOffset] = dA[offset + k - (2*k+1)*(last_row - 1 - l) - loadIdx - 1];
				sharedOffset += 20;
			}
			sharedOffset = 0;
			nextToAccess = last_row-1-20*i;
			__syncthreads();
		}

		tmp20 = dB[g_k * j + idx];
		
		if (nextToAccess < first_row) continue;
		tmp0 -= tmp20 * a_elements[sharedOffset]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp1 -= tmp20 * a_elements[sharedOffset+1]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp2 -= tmp20 * a_elements[sharedOffset+2]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp3 -= tmp20 * a_elements[sharedOffset+3]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp4 -= tmp20 * a_elements[sharedOffset+4]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp5 -= tmp20 * a_elements[sharedOffset+5]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp6 -= tmp20 * a_elements[sharedOffset+6]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp7 -= tmp20 * a_elements[sharedOffset+7]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp8 -= tmp20 * a_elements[sharedOffset+8]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp9 -= tmp20 * a_elements[sharedOffset+9]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp10 -= tmp20 * a_elements[sharedOffset+10]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp11 -= tmp20 * a_elements[sharedOffset+11]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp12 -= tmp20 * a_elements[sharedOffset+12]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp13 -= tmp20 * a_elements[sharedOffset+13]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp14 -= tmp20 * a_elements[sharedOffset+14]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp15 -= tmp20 * a_elements[sharedOffset+15]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp16 -= tmp20 * a_elements[sharedOffset+16]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp17 -= tmp20 * a_elements[sharedOffset+17]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp18 -= tmp20 * a_elements[sharedOffset+18]; nextToAccess--;

		if (nextToAccess < first_row) continue;
		tmp19 -= tmp20 * a_elements[sharedOffset+19]; nextToAccess--;
	}

	sharedOffset = 0;
	for (int j=row_to_start+1; j<row_to_start + 20 && j < last_row; j++) {
		int nextToAccess = last_row-1-20*i;
		for(int loadIdx = threadIdx.x+j-nextToAccess-1; loadIdx < j-nextToAccess+19 && loadIdx < k; loadIdx += step)
			a_elements[loadIdx-j+nextToAccess+1+sharedOffset] = dA[offset + k - (2*k+1)*(last_row - 1 - j) - loadIdx - 1];
		sharedOffset += 20;
	}
	sharedOffset = 0;
	__syncthreads();

	for (int j=row_to_start+1; j<row_to_start + 20 && j < last_row; j++, sharedOffset += 20) {
		int nextToAccess = last_row-1-20*i;

		tmp20 = dB[g_k * j + idx];

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp0 -= tmp20 * a_elements[sharedOffset]; nextToAccess--;
		
		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp1 -= tmp20 * a_elements[sharedOffset+1]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp2 -= tmp20 * a_elements[sharedOffset+2]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp3 -= tmp20 * a_elements[sharedOffset+3]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp4 -= tmp20 * a_elements[sharedOffset+4]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp5 -= tmp20 * a_elements[sharedOffset+5]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp6 -= tmp20 * a_elements[sharedOffset+6]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp7 -= tmp20 * a_elements[sharedOffset+7]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp8 -= tmp20 * a_elements[sharedOffset+8]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp9 -= tmp20 * a_elements[sharedOffset+9]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp10 -= tmp20 * a_elements[sharedOffset+10]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp11 -= tmp20 * a_elements[sharedOffset+11]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp12 -= tmp20 * a_elements[sharedOffset+12]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp13 -= tmp20 * a_elements[sharedOffset+13]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp14 -= tmp20 * a_elements[sharedOffset+14]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp15 -= tmp20 * a_elements[sharedOffset+15]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp16 -= tmp20 * a_elements[sharedOffset+16]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp17 -= tmp20 * a_elements[sharedOffset+17]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp18 -= tmp20 * a_elements[sharedOffset+18]; nextToAccess--;

		if (j - nextToAccess - 1 >= k || nextToAccess < first_row)
			continue;
		tmp19 -= tmp20 * a_elements[sharedOffset+19]; nextToAccess--;
	}

	row_to_start = last_row - 1 - 20 * i;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp0;
	for (int loadIdx = threadIdx.x; loadIdx < 19; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp1 -= tmp0 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp2 -= tmp0 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp3 -= tmp0 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp4 -= tmp0 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp5 -= tmp0 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp6 -= tmp0 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp7 -= tmp0 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp8 -= tmp0 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp9 -= tmp0 * a_elements[8];
	if (row_to_start - 10 >= first_row)
		tmp10 -= tmp0 * a_elements[9];
	if (row_to_start - 11 >= first_row)
		tmp11 -= tmp0 * a_elements[10];
	if (row_to_start - 12 >= first_row)
		tmp12 -= tmp0 * a_elements[11];
	if (row_to_start - 13 >= first_row)
		tmp13 -= tmp0 * a_elements[12];
	if (row_to_start - 14 >= first_row)
		tmp14 -= tmp0 * a_elements[13];
	if (row_to_start - 15 >= first_row)
		tmp15 -= tmp0 * a_elements[14];
	if (row_to_start - 16 >= first_row)
		tmp16 -= tmp0 * a_elements[15];
	if (row_to_start - 17 >= first_row)
		tmp17 -= tmp0 * a_elements[16];
	if (row_to_start - 18 >= first_row)
		tmp18 -= tmp0 * a_elements[17];
	if (row_to_start - 19 >= first_row)
		tmp19 -= tmp0 * a_elements[18];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp1;
	for (int loadIdx = threadIdx.x; loadIdx < 18; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp2 -= tmp1 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp3 -= tmp1 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp4 -= tmp1 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp5 -= tmp1 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp6 -= tmp1 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp7 -= tmp1 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp8 -= tmp1 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp9 -= tmp1 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp10 -= tmp1 * a_elements[8];
	if (row_to_start - 10 >= first_row)
		tmp11 -= tmp1 * a_elements[9];
	if (row_to_start - 11 >= first_row)
		tmp12 -= tmp1 * a_elements[10];
	if (row_to_start - 12 >= first_row)
		tmp13 -= tmp1 * a_elements[11];
	if (row_to_start - 13 >= first_row)
		tmp14 -= tmp1 * a_elements[12];
	if (row_to_start - 14 >= first_row)
		tmp15 -= tmp1 * a_elements[13];
	if (row_to_start - 15 >= first_row)
		tmp16 -= tmp1 * a_elements[14];
	if (row_to_start - 16 >= first_row)
		tmp17 -= tmp1 * a_elements[15];
	if (row_to_start - 17 >= first_row)
		tmp18 -= tmp1 * a_elements[16];
	if (row_to_start - 18 >= first_row)
		tmp19 -= tmp1 * a_elements[17];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp2;
	for (int loadIdx = threadIdx.x; loadIdx < 17; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp3 -= tmp2 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp4 -= tmp2 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp5 -= tmp2 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp6 -= tmp2 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp7 -= tmp2 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp8 -= tmp2 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp9 -= tmp2 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp10 -= tmp2 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp11 -= tmp2 * a_elements[8];
	if (row_to_start - 10 >= first_row)
		tmp12 -= tmp2 * a_elements[9];
	if (row_to_start - 11 >= first_row)
		tmp13 -= tmp2 * a_elements[10];
	if (row_to_start - 12 >= first_row)
		tmp14 -= tmp2 * a_elements[11];
	if (row_to_start - 13 >= first_row)
		tmp15 -= tmp2 * a_elements[12];
	if (row_to_start - 14 >= first_row)
		tmp16 -= tmp2 * a_elements[13];
	if (row_to_start - 15 >= first_row)
		tmp17 -= tmp2 * a_elements[14];
	if (row_to_start - 16 >= first_row)
		tmp18 -= tmp2 * a_elements[15];
	if (row_to_start - 17 >= first_row)
		tmp19 -= tmp2 * a_elements[16];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp3;
	for (int loadIdx = threadIdx.x; loadIdx < 16; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start- 1 >= first_row)
		tmp4 -= tmp3 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp5 -= tmp3 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp6 -= tmp3 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp7 -= tmp3 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp8 -= tmp3 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp9 -= tmp3 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp10 -= tmp3 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp11 -= tmp3 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp12 -= tmp3 * a_elements[8];
	if (row_to_start - 10 >= first_row)
		tmp13 -= tmp3 * a_elements[9];
	if (row_to_start - 11 >= first_row)
		tmp14 -= tmp3 * a_elements[10];
	if (row_to_start - 12 >= first_row)
		tmp15 -= tmp3 * a_elements[11];
	if (row_to_start - 13 >= first_row)
		tmp16 -= tmp3 * a_elements[12];
	if (row_to_start - 14 >= first_row)
		tmp17 -= tmp3 * a_elements[13];
	if (row_to_start - 15 >= first_row)
		tmp18 -= tmp3 * a_elements[14];
	if (row_to_start - 16 >= first_row)
		tmp19 -= tmp3 * a_elements[15];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp4;
	for (int loadIdx = threadIdx.x; loadIdx < 15; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp5 -= tmp4 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp6 -= tmp4 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp7 -= tmp4 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp8 -= tmp4 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp9 -= tmp4 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp10 -= tmp4 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp11 -= tmp4 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp12 -= tmp4 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp13 -= tmp4 * a_elements[8];
	if (row_to_start - 10 >= first_row)
		tmp14 -= tmp4 * a_elements[9];
	if (row_to_start - 11 >= first_row)
		tmp15 -= tmp4 * a_elements[10];
	if (row_to_start - 12 >= first_row)
		tmp16 -= tmp4 * a_elements[11];
	if (row_to_start - 13 >= first_row)
		tmp17 -= tmp4 * a_elements[12];
	if (row_to_start - 14 >= first_row)
		tmp18 -= tmp4 * a_elements[13];
	if (row_to_start - 15 >= first_row)
		tmp19 -= tmp4 * a_elements[14];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp5;
	for (int loadIdx = threadIdx.x; loadIdx < 14; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp6 -= tmp5 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp7 -= tmp5 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp8 -= tmp5 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp9 -= tmp5 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp10 -= tmp5 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp11 -= tmp5 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp12 -= tmp5 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp13 -= tmp5 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp14 -= tmp5 * a_elements[8];
	if (row_to_start - 10 >= first_row)
		tmp15 -= tmp5 * a_elements[9];
	if (row_to_start - 11 >= first_row)
		tmp16 -= tmp5 * a_elements[10];
	if (row_to_start - 12 >= first_row)
		tmp17 -= tmp5 * a_elements[11];
	if (row_to_start - 13 >= first_row)
		tmp18 -= tmp5 * a_elements[12];
	if (row_to_start - 14 >= first_row)
		tmp19 -= tmp5 * a_elements[13];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp6;
	for (int loadIdx = threadIdx.x; loadIdx < 13; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp7 -= tmp6 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp8 -= tmp6 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp9 -= tmp6 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp10 -= tmp6 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp11 -= tmp6 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp12 -= tmp6 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp13 -= tmp6 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp14 -= tmp6 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp15 -= tmp6 * a_elements[8];
	if (row_to_start - 10 >= first_row)
		tmp16 -= tmp6 * a_elements[9];
	if (row_to_start - 11 >= first_row)
		tmp17 -= tmp6 * a_elements[10];
	if (row_to_start - 12 >= first_row)
		tmp18 -= tmp6 * a_elements[11];
	if (row_to_start - 13 >= first_row)
		tmp19 -= tmp6 * a_elements[12];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp7;
	for (int loadIdx = threadIdx.x; loadIdx < 12; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp8 -= tmp7 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp9 -= tmp7 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp10 -= tmp7 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp11 -= tmp7 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp12 -= tmp7 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp13 -= tmp7 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp14 -= tmp7 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp15 -= tmp7 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp16 -= tmp7 * a_elements[8];
	if (row_to_start - 10 >= first_row)
		tmp17 -= tmp7 * a_elements[9];
	if (row_to_start - 11 >= first_row)
		tmp18 -= tmp7 * a_elements[10];
	if (row_to_start - 12 >= first_row)
		tmp19 -= tmp7 * a_elements[11];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp8;
	for (int loadIdx = threadIdx.x; loadIdx < 11; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp9 -= tmp8 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp10 -= tmp8 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp11 -= tmp8 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp12 -= tmp8 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp13 -= tmp8 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp14 -= tmp8 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp15 -= tmp8 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp16 -= tmp8 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp17 -= tmp8 * a_elements[8];
	if (row_to_start - 10 >= first_row)
		tmp18 -= tmp8 * a_elements[9];
	if (row_to_start - 11 >= first_row)
		tmp19 -= tmp8 * a_elements[10];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp9;
	for (int loadIdx = threadIdx.x; loadIdx < 10; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp10 -= tmp9 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp11 -= tmp9 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp12 -= tmp9 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp13 -= tmp9 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp14 -= tmp9 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp15 -= tmp9 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp16 -= tmp9 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp17 -= tmp9 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp18 -= tmp9 * a_elements[8];
	if (row_to_start - 10 >= first_row)
		tmp19 -= tmp9 * a_elements[9];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp10;
	for (int loadIdx = threadIdx.x; loadIdx < 9; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp11 -= tmp10 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp12 -= tmp10 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp13 -= tmp10 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp14 -= tmp10 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp15 -= tmp10 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp16 -= tmp10 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp17 -= tmp10 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp18 -= tmp10 * a_elements[7];
	if (row_to_start - 9 >= first_row)
		tmp19 -= tmp10 * a_elements[8];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp11;
	for (int loadIdx = threadIdx.x; loadIdx < 8; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp12 -= tmp11 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp13 -= tmp11 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp14 -= tmp11 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp15 -= tmp11 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp16 -= tmp11 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp17 -= tmp11 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp18 -= tmp11 * a_elements[6];
	if (row_to_start - 8 >= first_row)
		tmp19 -= tmp11 * a_elements[7];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp12;
	for (int loadIdx = threadIdx.x; loadIdx < 7; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp13 -= tmp12 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp14 -= tmp12 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp15 -= tmp12 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp16 -= tmp12 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp17 -= tmp12 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp18 -= tmp12 * a_elements[5];
	if (row_to_start - 7 >= first_row)
		tmp19 -= tmp12 * a_elements[6];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp13;
	for (int loadIdx = threadIdx.x; loadIdx < 6; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp14 -= tmp13 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp15 -= tmp13 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp16 -= tmp13 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp17 -= tmp13 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp18 -= tmp13 * a_elements[4];
	if (row_to_start - 6 >= first_row)
		tmp19 -= tmp13 * a_elements[5];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp14;
	for (int loadIdx = threadIdx.x; loadIdx < 5; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp15 -= tmp14 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp16 -= tmp14 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp17 -= tmp14 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp18 -= tmp14 * a_elements[3];
	if (row_to_start - 5 >= first_row)
		tmp19 -= tmp14 * a_elements[4];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp15;
	for (int loadIdx = threadIdx.x; loadIdx < 4; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp16 -= tmp15 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp17 -= tmp15 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp18 -= tmp15 * a_elements[2];
	if (row_to_start - 4 >= first_row)
		tmp19 -= tmp15 * a_elements[3];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp16;
	for (int loadIdx = threadIdx.x; loadIdx < 3; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp17 -= tmp16 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp18 -= tmp16 * a_elements[1];
	if (row_to_start - 3 >= first_row)
		tmp19 -= tmp16 * a_elements[2];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp17;
	for (int loadIdx = threadIdx.x; loadIdx < 2; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp18 -= tmp17 * a_elements[0];
	if (row_to_start - 2 >= first_row)
		tmp19 -= tmp17 * a_elements[1];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp18;
	for (int loadIdx = threadIdx.x; loadIdx < 1; loadIdx += step)
		a_elements[loadIdx] = dA[offset + k - (2*k+1)*(last_row - 1 - row_to_start) - loadIdx - 1];
	__syncthreads();
	if (row_to_start - 1 >= first_row)
		tmp19 -= tmp18 * a_elements[0];
	row_to_start--;

	if (row_to_start < first_row) return;
	dB[g_k * row_to_start + idx] = tmp19;
}

template <typename T>
__global__ void
fwdElim_spike(int N, int *ks, int g_k, int rightWidth, int *offsets, T *dA, T *dB, int partition_size, int rest_num, int *left_spike_widths, int *right_spike_widths, int *first_rows)
{
	__shared__ T a_elements[512];

	int k, offset, first_row, last_row, idx, bidy = blockIdx.y;

	if (bidy < gridDim.y / 2) {
		k = ks[bidy];
		offset = offsets[bidy];
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
		offset += (2*k+1) * (first_rows[bidy] - first_row);
		first_row = first_rows[bidy];
		fwdElim_offDiag_large_tiled(dA, dB, idx, k, g_k, right_spike_widths[bidy], first_row, last_row, offset, a_elements);
	} else {
		bidy -= gridDim.y/2 - 1;
		k = ks[bidy];
		offset = offsets[bidy];
		first_row = bidy*partition_size;
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= left_spike_widths[bidy-1]) return;
		idx += g_k - left_spike_widths[bidy-1];
		if (bidy < rest_num) {
			first_row += bidy;
			last_row = first_row + partition_size + 1;
		} else {
			first_row += rest_num;
			last_row = first_row + partition_size;
		}
		fwdElim_offDiag_large_tiled(dA, dB, idx, k, g_k, left_spike_widths[bidy-1], first_row, last_row, offset, a_elements);
	}
}

template <typename T>
__global__ void
bckElim_spike(int N, int *ks, int g_k, int rightWidth, int *offsets, T *dA, T *dB, int partition_size, int rest_num, int *left_spike_widths, int *right_spike_widths, int *first_rows)
{
	__shared__ T a_elements[512];

	int k, offset, first_row, last_row, idx, bidy = blockIdx.y;

	if (bidy < gridDim.y / 2) {
		k = ks[bidy];
		offset = offsets[bidy];
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
		offset += (2*k+1) * (last_row - 1 - first_row);
		first_row = first_rows[bidy];
		bckElim_offDiag_large_tiled(dA, dB, idx, k, g_k, right_spike_widths[bidy], first_row, last_row, offset, a_elements);
	} else {
		bidy -= gridDim.y/2 - 1;
		k = ks[bidy];
		offset = offsets[bidy];
		first_row = bidy*partition_size;
		idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= left_spike_widths[bidy-1]) return;
		idx += g_k - left_spike_widths[bidy-1];
		if (bidy < rest_num) {
			first_row += bidy;
			last_row = first_row + partition_size + 1;
		} else {
			first_row += rest_num;
			last_row = first_row + partition_size;
		}
		offset += (2*k+1) * (last_row - 1 - first_row);
		bckElim_offDiag_large_tiled(dA, dB, idx, k, g_k, left_spike_widths[bidy-1], first_row, last_row, offset, a_elements);
	}
}


// ----------------------------------------------------------------------------
// Forward/backward substitution kernels for calculating the right spikes.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
fwdElim_rightSpike_per_partition(int N, int k, int pivotIdx, T *dA, T *dB, int first_row, int last_row)
{
	int tid = threadIdx.x, bidx = blockIdx.x * N;
	if (tid >= k) return;
	int it_last = k;

	for(int i=first_row; i<last_row-k; i++) {
		T tmp = dB[bidx+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i+ttid+1] -= tmp * dA[pivotIdx + ttid + 1];
		pivotIdx += (k<<1)+1;
		__syncthreads();
	}
	for(int i=(first_row > last_row-k ? first_row : (last_row - k)); i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		it_last --;
		T tmp = dB[bidx+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i+ttid+1] -= tmp * dA[pivotIdx + ttid + 1];
		pivotIdx += (k<<1)+1;
		__syncthreads();
	}
}

template <typename T>
__global__ void
preBck_rightSpike_divide_per_partition(int N, int k, int pivotIdx, T *dA, T *dB, int first_row, int last_row)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (first_row + idx >= last_row) return;
	dB[blockIdx.y * N + first_row + idx] /= dA[pivotIdx + idx * ((k<<1)+1)];
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
preBck_offDiag_divide(int N, int g_k, int *ks, int *offsets, T *dA, T *dB, int partSize, int remainder) {
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
	int pivotIdx = offsets[partId] + k + (2*k+1)*(row_idx - first_row);

	dB[row_idx * g_k + idx] /= dA[pivotIdx];
}

template <typename T>
__global__ void
bckElim_rightSpike_per_partition(int N, int k, int pivotIdx, T *dA, T *dB, int first_row, int last_row)
{
	int tid = threadIdx.x, bidx = blockIdx.x * N;
	if (tid >= k) return;

	int it_last = k;

	for(int i=last_row-1; i>=first_row + k; i--) {
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i-ttid-1] -= dB[bidx+i] * dA[pivotIdx-ttid-1];
		pivotIdx -= (k<<1)+1;
		__syncthreads();
	}

	for(int i= (k-1+first_row > last_row - 1 ? last_row - 1 : k-1+first_row); i>=first_row; i--) {
		if(tid>=i-first_row) return;
		it_last --;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i-ttid-1] -= dB[bidx+i] * dA[pivotIdx - ttid - 1];
		pivotIdx -= (k<<1)+1;
		__syncthreads();
	}
}

// ----------------------------------------------------------------------------
// Forward/backward substitution kernels for calculating the left spikes.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void
fwdElim_leftSpike_per_partition(int N, int k, int bid_delta, int pivotIdx, T *dA, T *dB, int first_row, int last_row)
{
	int tid = threadIdx.x, bidx = (blockIdx.x+bid_delta) * N;
	if (tid >= k) return;

	int it_last = k;

	for(int i=first_row; i<last_row-k; i++) {
		T tmp = dB[bidx+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i+ttid+1] -= tmp * dA[pivotIdx + ttid + 1];
		pivotIdx += (k<<1)+1;
		__syncthreads();
	}
	for(int i=last_row-k; i<last_row-1; i++) {
		if(tid >= last_row-i-1) return;
		if(it_last > last_row-i-1)
			it_last = last_row-i-1;
		T tmp = dB[bidx+i];
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i+ttid+1] -= tmp * dA[pivotIdx + ttid + 1];
		pivotIdx += (k<<1)+1;
		__syncthreads();
	}
}

template <typename T>
__global__ void
preBck_leftSpike_divide_per_partition(int N, int k, int bid_delta, int pivotIdx, T *dA, T *dB, int first_row, int last_row)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (first_row + idx >= last_row) return;
	dB[(blockIdx.y+bid_delta) * N + first_row + idx] /= dA[pivotIdx + idx * ((k<<1)+1)];
}

template <typename T>
__global__ void
bckElim_leftSpike_per_partition(int N, int k, int bid_delta, int pivotIdx, T *dA, T *dB, int first_row, int last_row)
{
	int tid = threadIdx.x, bidx = (blockIdx.x+bid_delta) * N;
	if (tid >= k) return;

	int it_last = k;

	for(int i=last_row-1; i>=k+first_row; i--) {
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i-ttid-1] -= dB[bidx+i] * dA[pivotIdx-ttid-1];
		pivotIdx -= (k<<1)+1;
		__syncthreads();
	}
	for(int i=k-1+first_row; i>=first_row; i--) {
		if(tid>=i-first_row) return;
		it_last --;
		for(int ttid = tid; ttid<it_last; ttid+=blockDim.x)
			dB[bidx+i-ttid-1] -= dB[bidx+i] * dA[pivotIdx - ttid - 1];
		pivotIdx -= (k<<1)+1;
		__syncthreads();
	}
}


} // namespace var
} // namespace device
} // namespace spike


#endif

