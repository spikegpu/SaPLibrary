#ifndef INNER_PRODUCT_CUH
#define INNER_PRODUCT_CUH


namespace spike {
namespace device {


template <typename T>
__global__ void
innerProductBCX_g256(T*  d_spike,
                     T*  dB,
                     T*  dB_final,
                     int N,
                     int k,
                     int b_partition_size,
                     int b_partition_num,
                     int b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y, bidz = blockIdx.z;
	int idx = tid + bidx*k;

	volatile __shared__ T shared_inner[260];
	volatile __shared__ T shared_inner2[260];

	shared_inner[tid] = 0;
	shared_inner2[tid] = 0;

	int numThreads = blockDim.x;

	if(bidy+1 <= b_rest_num) {
		for(int ttid = tid, tmp_idx = idx; ttid < k; ttid += numThreads, tmp_idx += numThreads) {
			shared_inner[tid] += d_spike[tmp_idx + 2*k*k*bidy] * dB[bidz*N+(bidy+1)*(b_partition_size+1)+ttid];
			shared_inner2[tid] += d_spike[tmp_idx + (2*bidy+1)*k*k] * dB[bidz*N+(bidy+1)*(b_partition_size+1)-k+ttid];
		}
	} else {
		for(int ttid = tid, tmp_idx = idx; ttid < k; ttid += numThreads, tmp_idx += numThreads) {
			shared_inner[tid] += d_spike[tmp_idx + 2*k*k*bidy] * dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num+ttid];
			shared_inner2[tid] += d_spike[tmp_idx + (2*bidy+1)*k*k] * dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+ttid];
		}
	}

	__syncthreads();

	if(tid >= 64)
		return;

	T sum = shared_inner[tid], sum2 = shared_inner2[tid];
	for(int i=tid+64; i<numThreads; i+=64) {
		sum += shared_inner[i];
		sum2 += shared_inner2[i];
	}
	shared_inner[tid] = sum;
	shared_inner2[tid] = sum2;

	__syncthreads();

	if(tid >= 32)
		return;

	shared_inner[tid] += shared_inner[tid+32];
	shared_inner[tid] += shared_inner[tid+16];
	shared_inner[tid] += shared_inner[tid+8];
	shared_inner[tid] += shared_inner[tid+4];
	shared_inner[tid] += shared_inner[tid+2];

	shared_inner2[tid] += shared_inner2[tid+32];
	shared_inner2[tid] += shared_inner2[tid+16];
	shared_inner2[tid] += shared_inner2[tid+8];
	shared_inner2[tid] += shared_inner2[tid+4];
	shared_inner2[tid] += shared_inner2[tid+2];

	__syncthreads();

	if(tid > 0) return;

	if(bidy+1 <= b_rest_num) {
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)+bidx] -= shared_inner2[0] + shared_inner2[1];
	} else {
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num+bidx] -= shared_inner2[0] + shared_inner2[1];
	}
}


template <typename T>
__global__ void
innerProductBCX_g64(T*  d_spike,
                    T*  dB,
                    T*  dB_final,
                    int N,
                    int k,
                    int b_partition_size,
                    int b_partition_num,
                    int b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y, bidz = blockIdx.z;
	int idx = tid + bidx*k;

	volatile __shared__ T sharedB[256];
	volatile __shared__ T sharedB2[256];
	volatile __shared__ T shared_inner[256];
	volatile __shared__ T shared_inner2[256];

	sharedB[tid] = 0;
	sharedB2[tid] = 0;
	shared_inner[tid] = 0;
	shared_inner2[tid] = 0;

	if(tid >= k) return;

	if(bidy+1 <= b_rest_num) {
		sharedB[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)+tid];
	} else {
		sharedB[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num+tid];
	}

	shared_inner[tid] = d_spike[idx + 2*k*k*bidy] * sharedB2[tid];
	shared_inner2[tid] = d_spike[idx + (2*bidy+1)*k*k] * sharedB[tid];

	__syncthreads();

	if(tid >= 64)
		return;

	T sum = shared_inner[tid], sum2 = shared_inner2[tid];
	for(int i=tid+64; i<k; i+=64) {
		sum += shared_inner[i];
		sum2 += shared_inner2[i];
	}
	shared_inner[tid] = sum;
	shared_inner2[tid] = sum2;

	__syncthreads();

	if(tid >= 32)
		return;

	shared_inner[tid] += shared_inner[tid+32];
	shared_inner[tid] += shared_inner[tid+16];
	shared_inner[tid] += shared_inner[tid+8];
	shared_inner[tid] += shared_inner[tid+4];
	shared_inner[tid] += shared_inner[tid+2];

	shared_inner2[tid] += shared_inner2[tid+32];
	shared_inner2[tid] += shared_inner2[tid+16];
	shared_inner2[tid] += shared_inner2[tid+8];
	shared_inner2[tid] += shared_inner2[tid+4];
	shared_inner2[tid] += shared_inner2[tid+2];

	__syncthreads();

	if(tid > 0) return;

	if(bidy+1 <= b_rest_num) {
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)+bidx] -= shared_inner2[0] + shared_inner2[1];
	} else {
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num+bidx] -= shared_inner2[0] + shared_inner2[1];
	}
}


template <typename T>
__global__ void
innerProductBCX_g32(T*  d_spike,
                    T*  dB,
                    T*  dB_final,
                    int N,
                    int k,
                    int b_partition_size,
                    int b_partition_num,
                    int b_rest_num) 
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y, bidz = blockIdx.z;
	int idx = tid + bidx*k;

	volatile __shared__ T sharedB[100];
	volatile __shared__ T sharedB2[100];
	volatile __shared__ T shared_inner[100];
	volatile __shared__ T shared_inner2[100];

	sharedB[tid] = 0;
	sharedB2[tid] = 0;
	shared_inner[tid] = 0;
	shared_inner2[tid] = 0;

	if(tid >= k) return;
	if(bidy+1 <= b_rest_num) {
		sharedB[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)+tid];
	} else {
		sharedB[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num+tid];
	}

	shared_inner[tid] = d_spike[idx + 2*k*k*bidy] * sharedB2[tid];
	shared_inner2[tid] = d_spike[idx + (2*bidy+1)*k*k] * sharedB[tid];

	__syncthreads();

	if(tid >= 32) {
		shared_inner[tid-32] += shared_inner[tid];
		shared_inner2[tid-32] += shared_inner2[tid];
		return;
	}
	__syncthreads();

	shared_inner[tid] += shared_inner[tid+16];
	shared_inner[tid] += shared_inner[tid+8];
	shared_inner[tid] += shared_inner[tid+4];
	shared_inner[tid] += shared_inner[tid+2];

	shared_inner2[tid] += shared_inner2[tid+16];
	shared_inner2[tid] += shared_inner2[tid+8];
	shared_inner2[tid] += shared_inner2[tid+4];
	shared_inner2[tid] += shared_inner2[tid+2];

	__syncthreads();

	if(tid > 0) return;

	if(bidy+1 <= b_rest_num) {
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)+bidx] -= shared_inner2[0] + shared_inner2[1];
	} else {
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num+bidx] -= shared_inner2[0] + shared_inner2[1];
	}
}

template <typename T>
__global__ void
innerProductBCX(T*  d_spike,
                T*  dB,
                T*  dB_final,
                int N,
                int k,
                int b_partition_size,
                int b_partition_num,
                int b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y, bidz = blockIdx.z;
	int idx = tid + bidx*k;

	volatile __shared__ T sharedB[60];
	volatile __shared__ T sharedB2[60];
	volatile __shared__ T shared_inner[60];
	volatile __shared__ T shared_inner2[60];

	sharedB[tid] = 0;
	sharedB2[tid] = 0;
	shared_inner[tid] = 0;
	shared_inner2[tid] = 0;

	if(tid >= k) return;
	if(bidy+1 <= b_rest_num) {
		sharedB[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)+tid];
	} else {
		sharedB[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num+tid];
	}

	shared_inner[tid] = d_spike[idx + 2*k*k*bidy] * sharedB2[tid];
	shared_inner2[tid] = d_spike[idx + (2*bidy+1)*k*k] * sharedB[tid];

	shared_inner[tid] += shared_inner[tid+16];
	shared_inner[tid] += shared_inner[tid+8];
	shared_inner[tid] += shared_inner[tid+4];
	shared_inner[tid] += shared_inner[tid+2];

	shared_inner2[tid] += shared_inner2[tid+16];
	shared_inner2[tid] += shared_inner2[tid+8];
	shared_inner2[tid] += shared_inner2[tid+4];
	shared_inner2[tid] += shared_inner2[tid+2];

	__syncthreads();

	if(tid > 0) return;

	if(bidy+1 <= b_rest_num) {
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)+bidx] -= shared_inner2[0] + shared_inner2[1];
	} else {
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num+bidx] -= shared_inner2[0] + shared_inner2[1];
	}
}

template <typename T>
__global__ void
innerProductBCX_var_bandwidth_g256(T* d_spike,
                                   T*   dB,
                                   T*   dB_final,
                                   int  N,
                                   int* ks,
                                   int* offsets,
                                   int  b_partition_size,
                                   int  b_partition_num,
                                   int  b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y, bidz = blockIdx.z;
	int k = ks[bidy];
	if (bidx >= k) return;
	int offset1 = offsets[bidy];
	int offset2 = offsets[bidy]+k*k;
	int idx = tid + bidx*k;

	volatile __shared__ T shared_inner[260];
	volatile __shared__ T shared_inner2[260];

	shared_inner[tid] = 0;
	shared_inner2[tid] = 0;

	int numThreads = blockDim.x;

	if(bidy+1 <= b_rest_num) {
		for(int ttid = tid, tmp_idx = idx; ttid < k; ttid += numThreads, tmp_idx += numThreads) {
			shared_inner[tid] += d_spike[tmp_idx+offset1] * dB[bidz*N+(bidy+1)*(b_partition_size+1)+ttid];
			shared_inner2[tid] += d_spike[tmp_idx + offset2] * dB[bidz*N+(bidy+1)*(b_partition_size+1)-k+ttid];
		}
	} else {
		for(int ttid = tid, tmp_idx = idx; ttid < k; ttid += numThreads, tmp_idx += numThreads) {
			shared_inner[tid] += d_spike[tmp_idx + offset1] * dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num+ttid];
			shared_inner2[tid] += d_spike[tmp_idx + offset2] * dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+ttid];
		}
	}

	__syncthreads();

	if(tid >= 64)
		return;

	T sum = shared_inner[tid], sum2 = shared_inner2[tid];
	for(int i=tid+64; i<numThreads; i+=64) {
		sum += shared_inner[i];
		sum2 += shared_inner2[i];
	}
	shared_inner[tid] = sum;
	shared_inner2[tid] = sum2;

	__syncthreads();

	if(tid >= 32)
		return;

	shared_inner[tid] += shared_inner[tid+32];
	shared_inner[tid] += shared_inner[tid+16];
	shared_inner[tid] += shared_inner[tid+8];
	shared_inner[tid] += shared_inner[tid+4];
	shared_inner[tid] += shared_inner[tid+2];

	shared_inner2[tid] += shared_inner2[tid+32];
	shared_inner2[tid] += shared_inner2[tid+16];
	shared_inner2[tid] += shared_inner2[tid+8];
	shared_inner2[tid] += shared_inner2[tid+4];
	shared_inner2[tid] += shared_inner2[tid+2];

	__syncthreads();

	if(tid > 0) return;

	if(bidy+1 <= b_rest_num) {
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)+bidx] -= shared_inner2[0] + shared_inner2[1];
	} else {
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num+bidx] -= shared_inner2[0] + shared_inner2[1];
	}
}


template <typename T>
__global__ void
innerProductBCX_var_bandwidth_g64(T*   d_spike,
                                  T*   dB,
                                  T*   dB_final,
                                  int  N,
                                  int* ks,
                                  int* offsets,
                                  int  b_partition_size,
                                  int  b_partition_num,
                                  int  b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y, bidz = blockIdx.z;
	int k = ks[bidy];
	if (bidx >= k) return;
	int offset1 = offsets[bidy];
	int offset2 = offsets[bidy]+k*k;
	int idx = tid + bidx*k;

	volatile __shared__ T sharedB[256];
	volatile __shared__ T sharedB2[256];
	volatile __shared__ T shared_inner[256];
	volatile __shared__ T shared_inner2[256];

	sharedB[tid] = 0;
	sharedB2[tid] = 0;
	shared_inner[tid] = 0;
	shared_inner2[tid] = 0;

	if(tid >= k) return;

	if(bidy+1 <= b_rest_num) {
		sharedB[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)+tid];
	} else {
		sharedB[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num+tid];
	}

	shared_inner[tid] = d_spike[idx + offset1] * sharedB2[tid];
	shared_inner2[tid] = d_spike[idx + offset2] * sharedB[tid];

	__syncthreads();

	if(tid >= 64)
		return;

	T sum = shared_inner[tid], sum2 = shared_inner2[tid];
	for(int i=tid+64; i<k; i+=64) {
		sum += shared_inner[i];
		sum2 += shared_inner2[i];
	}
	shared_inner[tid] = sum;
	shared_inner2[tid] = sum2;

	__syncthreads();

	if(tid >= 32)
		return;

	shared_inner[tid] += shared_inner[tid+32];
	shared_inner[tid] += shared_inner[tid+16];
	shared_inner[tid] += shared_inner[tid+8];
	shared_inner[tid] += shared_inner[tid+4];
	shared_inner[tid] += shared_inner[tid+2];

	shared_inner2[tid] += shared_inner2[tid+32];
	shared_inner2[tid] += shared_inner2[tid+16];
	shared_inner2[tid] += shared_inner2[tid+8];
	shared_inner2[tid] += shared_inner2[tid+4];
	shared_inner2[tid] += shared_inner2[tid+2];

	__syncthreads();

	if(tid > 0) return;

	if(bidy+1 <= b_rest_num) {
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)+bidx] -= shared_inner2[0] + shared_inner2[1];
	} else {
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num+bidx] -= shared_inner2[0] + shared_inner2[1];
	}
}


template <typename T>
__global__ void
innerProductBCX_var_bandwidth_g32(T*   d_spike,
                                  T*   dB,
                                  T*   dB_final,
                                  int  N,
                                  int* ks,
                                  int* offsets,
                                  int  b_partition_size,
                                  int  b_partition_num,
                                  int  b_rest_num) 
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y, bidz = blockIdx.z;
	int k = ks[bidy];
	if (bidx >= k) return;
	int offset1 = offsets[bidy];
	int offset2 = offsets[bidy] + k*k;
	int idx = tid + bidx*k;

	volatile __shared__ T sharedB[100];
	volatile __shared__ T sharedB2[100];
	volatile __shared__ T shared_inner[100];
	volatile __shared__ T shared_inner2[100];

	sharedB[tid] = 0;
	sharedB2[tid] = 0;
	shared_inner[tid] = 0;
	shared_inner2[tid] = 0;

	if(tid >= k) return;
	if(bidy+1 <= b_rest_num) {
		sharedB[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)+tid];
	} else {
		sharedB[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num+tid];
	}

	shared_inner[tid] = d_spike[idx + offset1] * sharedB2[tid];
	shared_inner2[tid] = d_spike[idx + offset2] * sharedB[tid];

	__syncthreads();

	if(tid >= 32) {
		shared_inner[tid-32] += shared_inner[tid];
		shared_inner2[tid-32] += shared_inner2[tid];
		return;
	}
	__syncthreads();

	shared_inner[tid] += shared_inner[tid+16];
	shared_inner[tid] += shared_inner[tid+8];
	shared_inner[tid] += shared_inner[tid+4];
	shared_inner[tid] += shared_inner[tid+2];

	shared_inner2[tid] += shared_inner2[tid+16];
	shared_inner2[tid] += shared_inner2[tid+8];
	shared_inner2[tid] += shared_inner2[tid+4];
	shared_inner2[tid] += shared_inner2[tid+2];

	__syncthreads();

	if(tid > 0) return;

	if(bidy+1 <= b_rest_num) {
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)+bidx] -= shared_inner2[0] + shared_inner2[1];
	} else {
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num+bidx] -= shared_inner2[0] + shared_inner2[1];
	}
}

template <typename T>
__global__ void
innerProductBCX_var_bandwidth(T*   d_spike,
                              T*   dB,
                              T*   dB_final,
                              int  N,
                              int* ks,
                              int* offsets,
                              int  b_partition_size,
                              int  b_partition_num,
                              int  b_rest_num)
{
	int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y, bidz = blockIdx.z;
	int k = ks[bidy];
	if (bidx >= k) return;
	int offset1 = offsets[bidy];
	int offset2 = offsets[bidy] + k*k;
	int idx = tid + bidx*k;

	volatile __shared__ T sharedB[60];
	volatile __shared__ T sharedB2[60];
	volatile __shared__ T shared_inner[60];
	volatile __shared__ T shared_inner2[60];

	sharedB[tid] = 0;
	sharedB2[tid] = 0;
	shared_inner[tid] = 0;
	shared_inner2[tid] = 0;

	if(tid >= k) return;
	if(bidy+1 <= b_rest_num) {
		sharedB[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*(b_partition_size+1)+tid];
	} else {
		sharedB[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+tid];
		sharedB2[tid] = dB[bidz*N+(bidy+1)*b_partition_size+b_rest_num+tid];
	}

	shared_inner[tid] = d_spike[idx + offset1] * sharedB2[tid];
	shared_inner2[tid] = d_spike[idx + offset2] * sharedB[tid];

	shared_inner[tid] += shared_inner[tid+16];
	shared_inner[tid] += shared_inner[tid+8];
	shared_inner[tid] += shared_inner[tid+4];
	shared_inner[tid] += shared_inner[tid+2];

	shared_inner2[tid] += shared_inner2[tid+16];
	shared_inner2[tid] += shared_inner2[tid+8];
	shared_inner2[tid] += shared_inner2[tid+4];
	shared_inner2[tid] += shared_inner2[tid+2];

	__syncthreads();

	if(tid > 0) return;

	if(bidy+1 <= b_rest_num) {
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*(b_partition_size+1)+bidx] -= shared_inner2[0] + shared_inner2[1];
	} else {
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num-k+bidx] -= shared_inner[0] + shared_inner[1];
		dB_final[bidz*N+(bidy+1)*b_partition_size+b_rest_num+bidx] -= shared_inner2[0] + shared_inner2[1];
	}
}


} // namespace device
} // namespace spike


#endif
