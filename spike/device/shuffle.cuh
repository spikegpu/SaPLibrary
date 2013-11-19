#ifndef SHUFFLE_CUH
#define SHUFFLE_CUH



namespace spike {
namespace device {


template <typename T>
__global__ void
permute(int  N, 
        T*   ori_array,
        T*   final_array,
        int* per_array)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N)
		return;
	final_array[per_array[idx] + blockIdx.y*N] = ori_array[idx + blockIdx.y * N];
}

template <typename T>
__global__ void
columnPermute(int  N,
			  int  g_k,
			  T*   ori_array,
			  T*   final_array,
			  int* per_array)
{
	int idx = blockIdx.y + gridDim.y * blockIdx.z;
	if (idx >= N) return;
	int col_idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (col_idx >= g_k) return;
	final_array[per_array[idx]*g_k + col_idx] = ori_array[idx*g_k + col_idx];
}



} // namespace device
} // namespace spike


#endif
