/** \file data_transfer.cuh
 *
 * Data transfer CUDA kernels.
 */

#ifndef DATA_TRANSFER_CUH
#define DATA_TRANSFER_CUH


namespace sap {
namespace device {


template <typename T>
__global__ void
assembleReducedMat(int k,
                   T*  dWV,
                   T*  d_comp)
{
    int tid = threadIdx.x, bid = blockIdx.x;
    int r = tid/k, c = tid%k;
    int offset = bid * 4*k*k;
    if(r == c) {
        d_comp[2*k*(r+k) + c+k + offset] = 1.0;
        d_comp[2*k*r+c+offset] = 1.0;
    } else {
        d_comp[2*k*(r+k) + c+k+offset] = 0.0;
        d_comp[2*k*r+c+offset] = 0.0;
    }
    d_comp[2*k*r+c+k+offset] = dWV[k*(r+k)+c+2*k*k*bid];
    d_comp[2*k*(r+k)+c+offset] = dWV[k*r+c+2*k*k*bid];
}


template <typename T>
__global__ void
assembleReducedMat_g32(int k,
                       T*  dWV,
                       T*  d_comp)
{
    int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
    int k_square = k*k;
    int offset = bidy * 4*k_square;
    int r = bidx, c = tid;

    if(r == c) {
        d_comp[2*k*(r+k) + c+k + offset] = 1.0;
        d_comp[2*k*r+c+offset] = 1.0;
    } else {
        d_comp[2*k*(r+k) + c+k+offset] = 0.0;
        d_comp[2*k*r+c+offset] = 0.0;
    }
    d_comp[2*k*r+c+k+offset] = dWV[k*(r+k)+c+2*k_square*bidy];
    d_comp[2*k*(r+k)+c+offset] = dWV[k*r+c+2*k_square*bidy];
}


template <typename T>
__global__ void
assembleReducedMat_general(int k,
                           T*  dWV,
                           T*  d_comp)
{
    int tid = threadIdx.x, bidy = blockIdx.y;
    int k_square = k*k;
    int offset = bidy * 4*k_square;
    int r = blockIdx.x, c;

    for(c = tid; c < k; c+=blockDim.x) {
        if(r == c) {
            d_comp[2*k*(r+k) + c+k + offset] = 1.0;
            d_comp[2*k*r+c+offset] = 1.0;
        } else {
            d_comp[2*k*(r+k) + c+k+offset] = 0.0;
            d_comp[2*k*r+c+offset] = 0.0;
        }
        d_comp[2*k*r+c+k+offset] = dWV[k*(r+k)+c+2*k_square*bidy];
        d_comp[2*k*(r+k)+c+offset] = dWV[k*r+c+2*k_square*bidy];
    }
}


template <typename T>
__global__ void
copydAtodA2(int  N,
            int  k,
            T*   dA,
            T*   dA2,
            int  num_of_rows,
            int  partition_size,
            int  partition_num,
            int  rest_num)
{
    int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
    int two_k = (k<<1);
    if(bidy+1 <= rest_num)
        dA2[tid+(bidx+bidy*num_of_rows)*(two_k+1)] = dA[tid+(bidx + (bidy+1)*(partition_size+1))*(two_k+1)];
    else
        dA2[tid+(bidx+bidy*num_of_rows)*(two_k+1)] = dA[tid+(bidx + (bidy+1)*partition_size+rest_num)*(two_k+1)];
}


template <typename T>
__global__ void
copydAtodA2_general(int  N,
                    int  k,
                    T*   dA,
                    T*   dA2,
                    int  num_of_rows,
                    int  partition_size,
                    int  partition_num,
                    int  rest_num)
{
    int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
    int two_k = (k<<1);

    if(bidy+1 <= rest_num) {
        for(;tid <= two_k;  tid+=blockDim.x)
            dA2[tid+(bidx+bidy*num_of_rows)*(two_k+1)] = dA[tid+(bidx+(bidy+1)*(partition_size+1))*(two_k+1)];
    } else {
        for(;tid <= two_k;  tid+=blockDim.x)
            dA2[tid+(bidx+bidy*num_of_rows)*(two_k+1)] = dA[tid+(bidx+(bidy+1)*partition_size+rest_num)*(two_k+1)];
    }
}


template <typename T>
__global__ void
copydWV_general(int k,
                T*  dA,
                T*  dWV,
                T*  d_spike,
                int partition_size,
                int partition_num,
                int rest_num)
{
    int tid = threadIdx.x, r = blockIdx.x, bidy = blockIdx.y;
    int k_square = k*k;
    int c;

    if(bidy + 1 <= rest_num) {
        for(c = tid; c<k; c+=blockDim.x) {
            d_spike[c*k+r+2*bidy*k_square] = dWV[r*k+c+ 2*bidy*k_square] = (r > c ? 0.0 : dA[((bidy+1)*(partition_size+1)+r)*(2*k+1)+c-r]); // V/B matrix
            d_spike[c*k+r+(2*bidy+1)*k_square] = dWV[r*k+c+ (2*bidy+1)*k_square] =  (c > r ? 0.0 : dA[((bidy+1)*(partition_size+1)-k+r)*(2*k+1)+2*k-r+c]); // W/C matrix
        }
    } else {
        for(c = tid; c<k; c+=blockDim.x) {
            d_spike[c*k+r+2*bidy*k_square] = dWV[r*k+c+ 2*bidy*k_square] = (r > c ? 0.0 : dA[((bidy+1)*partition_size+rest_num+r)*(2*k+1)+c-r]); // V/B matrix
            d_spike[c*k+r+(2*bidy+1)*k_square] = dWV[r*k+c+ (2*bidy+1)*k_square] =  (c > r ? 0.0 : dA[((bidy+1)*partition_size+rest_num-k+r)*(2*k+1)+2*k-r+c]); // W/C matrix
        }
    }
}


template <typename T>
__global__ void
copydWV_g32(int k,
            T*  dA,
            T*  dWV,
            T*  d_spike,
            int partition_size,
            int partition_num,
            int rest_num)
{
    int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
    int k_square = k*k;
    int r = bidx, c = tid;

    if(bidy + 1 <= rest_num) {
        d_spike[c*k+r+2*bidy*k_square] = dWV[r*k+c+ 2*bidy*k_square] = (r > c ? 0.0 : dA[((bidy+1)*(partition_size+1)+r)*(2*k+1)+c-r]); // V/B matrix
        d_spike[c*k+r+(2*bidy+1)*k_square] = dWV[r*k+c+ (2*bidy+1)*k_square] =  (c > r ? 0.0 : dA[((bidy+1)*(partition_size+1)-k+r)*(2*k+1)+2*k-r+c]); // W/C matrix
    } else {
        d_spike[c*k+r+2*bidy*k_square] = dWV[r*k+c+ 2*bidy*k_square] = (r > c ? 0.0 : dA[((bidy+1)*partition_size+rest_num+r)*(2*k+1)+c-r]); // V/B matrix
        d_spike[c*k+r+(2*bidy+1)*k_square] = dWV[r*k+c+ (2*bidy+1)*k_square] =  (c > r ? 0.0 : dA[((bidy+1)*partition_size+rest_num-k+r)*(2*k+1)+2*k-r+c]); // W/C matrix
    }
}


template <typename T>
__global__ void
copydWV(int k,
        T*  dA,
        T*  dWV,
        T*  d_spike,
        int partition_size,
        int partition_num,
        int rest_num)
{
    int tid = threadIdx.x, bid = blockIdx.x;
    int k_square = k*k;
    int r = tid/k, c = tid%k;
    if(bid + 1 <= rest_num) {
        d_spike[c*k+r+2*bid*k_square] = dWV[tid + 2*bid*k_square] = (r > c ? 0.0 : dA[((bid+1)*(partition_size+1)+r)*(2*k+1)+c-r]); // V/B matrix
        d_spike[c*k+r+(2*bid+1)*k_square] = dWV[tid + (2*bid+1)*k_square] =  (c > r ? 0.0 : dA[((bid+1)*(partition_size+1)-k+r)*(2*k+1)+2*k-r+c]); // W/C matrix
    } else {
        d_spike[c*k+r+2*bid*k_square] = dWV[tid + 2*bid*k_square] = (r > c ? 0.0 : dA[((bid+1)*partition_size+rest_num+r)*(2*k+1)+c-r]); // V/B matrix
        d_spike[c*k+r+(2*bid+1)*k_square] = dWV[tid + (2*bid+1)*k_square] =  (c > r ? 0.0 : dA[((bid+1)*partition_size+rest_num-k+r)*(2*k+1)+2*k-r+c]); // W/C matrix
    }
}


template <typename T>
__global__ void
copydAtoPartialA(int N,
                 int k,
                 T*  dA,
                 T*  dA2,
                 T*  d_partial_A,
                 int partition_size,
                 int partition_num,
                 int rest_num,
                 int num_of_rows)
{
    int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
    if(bidy * 2 < gridDim.y) {
        if(bidy + 1 <= rest_num) {
            d_partial_A[(2*bidy*(k+1)+bidx)*(2*k+1)+tid] = dA[((bidy+1)*(partition_size+1)-k+bidx)*(2*k+1)+tid]; 
        } else {
            d_partial_A[(2*bidy*(k+1)+bidx)*(2*k+1)+tid] = dA[((bidy+1)*partition_size+rest_num-k+bidx)*(2*k+1)+tid]; 
        }
    }
    else {
        bidy -= gridDim.y/2;
        d_partial_A[((2*bidy+1)*(k+1)+bidx)*(2*k+1)+tid] = dA2[(bidy*num_of_rows+bidx)*(2*k+1)+tid];
    }
}


template <typename T>
__global__ void
copydAtoPartialA_general(int N,
                         int k,
                         T*  dA,
                         T*  dA2,
                         T*  d_partial_A,
                         int partition_size,
                         int partition_num,
                         int rest_num,
                         int num_of_rows)
{
    int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
    int two_k_plus_1 = 2*k+1;
    int step = blockDim.x;
    if(bidy * 2 < gridDim.y) {
        if(bidy + 1 <= rest_num) {
            for(; tid < two_k_plus_1; tid+=step)
                d_partial_A[(2*bidy*(k+1)+bidx)*(2*k+1)+tid] = dA[((bidy+1)*(partition_size+1)-k+bidx)*(2*k+1)+tid];
        } else {
            for(; tid < two_k_plus_1; tid+=step)
                d_partial_A[(2*bidy*(k+1)+bidx)*(2*k+1)+tid] = dA[((bidy+1)*partition_size+rest_num-k+bidx)*(2*k+1)+tid];
        }
    }
    else {
        bidy -= gridDim.y/2;
        for(; tid < two_k_plus_1; tid+=step)
            d_partial_A[((2*bidy+1)*(k+1)+bidx)*(2*k+1)+tid] = dA2[(bidy*num_of_rows+bidx)*(2*k+1)+tid];
    }
}


template <typename T>
__global__ void
copyWVFromOrToExtendedV(int N,
                        int k,
                        int partition_size,
                        int rest_num,
                        T*  dWV,
                        T*  d_eV,
bool from)
{
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int offset = 0, my_partition_size = 0;
    if(bidy < rest_num) {
        offset = (partition_size+1) * bidy;
        my_partition_size = partition_size + 1;
    }
    else {
        offset = partition_size * bidy + rest_num;
        my_partition_size = partition_size;
    }
    if (!from)
        d_eV[my_partition_size-k+threadIdx.x+offset+bidx*N] = dWV[threadIdx.x+bidx*k+2*k*k*bidy];
    else
        dWV[threadIdx.x+bidx*k+2*k*k*bidy] = d_eV[my_partition_size-k+threadIdx.x+offset+bidx*N];
}


template <typename T>
__global__ void
copyWVFromOrToExtendedV_general(int  N,
                                int  k,
                                int  partition_size,
                                int  rest_num,
                                T*   dWV,
                                T*   d_eV,
                                bool from)
{
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int offset = 0, my_partition_size = 0;
    if(bidy < rest_num) {
        offset = (partition_size+1) * bidy;
        my_partition_size = partition_size + 1;
    }
    else {
        offset = partition_size * bidy + rest_num;
        my_partition_size = partition_size;
    }

    if (!from) {
        for(int ttid=threadIdx.x; ttid < k; ttid += blockDim.x)
            d_eV[my_partition_size-k+ttid+offset+bidx*N] = dWV[ttid+bidx*k+2*k*k*bidy];
    } else {
        for(int ttid=threadIdx.x; ttid < k; ttid += blockDim.x)
            dWV[ttid+bidx*k+2*k*k*bidy] = d_eV[my_partition_size-k+ttid+offset+bidx*N];
    }
}

template <typename T>
__global__ void
copyWVFromOrToExtendedWVTranspose_general(int  row_size,
                                          int  k,
                                          int  rightWidth,
                                          int  partition_size,
                                          int  rest_num,
                                          int  column_deltaW,
                                          T*   dWV,
                                          T*   d_eWV,
                                          bool from)
{
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int offset = 0, my_partition_size = 0;
    if(bidy < rest_num) {
        offset = (partition_size+1) * bidy;
        my_partition_size = partition_size + 1;
    }
    else {
        offset = partition_size * bidy + rest_num;
        my_partition_size = partition_size;
    }

    if (!from) {
        if (bidy < gridDim.y - 1 && bidx < rightWidth) {
            for(int ttid=threadIdx.x; ttid < k; ttid += blockDim.x) {
                d_eWV[(my_partition_size-k+ttid+offset)*row_size+bidx] = dWV[ttid+bidx*k+2*k*k*bidy];
            }
        }
        if (bidy > 0 && bidx >= rightWidth) {
            for(int ttid=threadIdx.x; ttid < k; ttid += blockDim.x) {
                d_eWV[(ttid+offset)*row_size+bidx] = dWV[ttid+(bidx+column_deltaW)*k+k*k*(2*(bidy-1)+1)];
            }
        }
    } else {
        if (bidy < gridDim.y - 1 && bidx < rightWidth) {
            for(int ttid=threadIdx.x; ttid < k; ttid += blockDim.x) {
                 dWV[ttid+bidx*k+2*k*k*bidy] = d_eWV[(my_partition_size-k+ttid+offset)*row_size+bidx];
            }
        }
        if (bidy > 0 && bidx >= rightWidth) {
            for(int ttid=threadIdx.x; ttid < k; ttid += blockDim.x) {
                 dWV[ttid+(bidx+column_deltaW)*k+k*k*(2*(bidy-1)+1)] = d_eWV[(ttid+offset)*row_size+bidx];
            }
        }
    }
}

template <typename T>
__global__ void
copyWVFromOrToExtendedW(int  N,
                        int  k,
                        int  partition_size,
                        int  rest_num,
                        T*   dWV,
                        T*   d_eW,
                        bool from)
{
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int offset = 0;
    if(bidy < rest_num) {
        offset = (partition_size+1) * bidy;
    }
    else {
        offset = partition_size * bidy + rest_num;
    }

    if (!from) 
        d_eW[threadIdx.x+offset+bidx*N] = dWV[threadIdx.x+bidx*k+k*k*(2*bidy+1)];
    else
        dWV[threadIdx.x+bidx*k+k*k*(2*bidy+1)] = d_eW[threadIdx.x+offset+bidx*N];
}


template <typename T>
__global__ void
copyWVFromOrToExtendedW_general(int  N,
                                int  k,
                                int  partition_size,
                                int  rest_num,
                                T*   dWV,
                                T*   d_eW,
                                bool from)
{
    int bidx = blockIdx.x, bidy = blockIdx.y;
    int offset = 0;
    if(bidy < rest_num) {
        offset = (partition_size+1) * bidy;
    }
    else {
        offset = partition_size * bidy + rest_num;
    }

    if (!from) {
        for(int ttid=threadIdx.x; ttid < k; ttid += blockDim.x)
            d_eW[ttid+offset+bidx*N] = dWV[ttid+bidx*k+k*k*(2*bidy+1)];
    } else {
        for(int ttid=threadIdx.x; ttid < k; ttid += blockDim.x)
            dWV[ttid+bidx*k+k*k*(2*bidy+1)] = d_eW[ttid+offset+bidx*N];
    }
}


template <typename T>
__global__ void
copyFromCOOMatrixToBandedMatrix(int  nnz,
                                int  bandwidth,
                                int* rows,
                                int* cols,
                                T*   vals,
                                T*   dB,
                                int  row_num_bias,
                                bool saveMem)
{
    int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
    int idx = tid + bidx * blockDim.x + bidy * gridDim.x * blockDim.x;
    if(idx >= nnz) return;

    int j = rows[idx] - row_num_bias, l = cols[idx] - row_num_bias;
    if (saveMem && j < l)
        return;

    int col_width = 2 * bandwidth + 1;
    int delta = bandwidth;
    if (saveMem) {
        col_width = bandwidth + 1;
        delta = 0;
    }

    dB[l * col_width + delta + j - l] = vals[idx];
}

namespace var {

template <typename T>
__global__ void
copyFromCOOMatrixToBandedMatrix(int  nnz,
                                int* ks,
                                int* rows,
                                int* cols,
                                T*   vals,
                                T*   dB,
                                int* offsets,
                                int  partSize,
                                int  remainder,
                                int  row_num_bias,
                                bool saveMem)
{
    int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
    int idx = tid + bidx * blockDim.x + bidy * gridDim.x * blockDim.x;
    if(idx >= nnz) return;

    int j = rows[idx] - row_num_bias, l = cols[idx] - row_num_bias;

    if (saveMem && j < l)
        return;

    int curPartNum = l / (partSize + 1);
    int l_in_part;
    if (curPartNum >= remainder) {
        l_in_part = l - remainder * (partSize + 1);
        curPartNum = remainder + l_in_part / partSize;
        l_in_part %= partSize;
    } else {
        l_in_part = l % (partSize + 1);
    }

    int bandwidth = ks[curPartNum];

    int col_width = 2 * bandwidth + 1;
    int delta = bandwidth;
    if (saveMem) {
        col_width = bandwidth + 1;
        delta = 0;
    }

    dB[offsets[curPartNum] + l_in_part * col_width + delta + j - l] = vals[idx];
}

template <typename T>
__global__ void
assembleReducedMat(int* ks,
                   int* offsets_src,
                   int* offsets_dst,
                   T*   dWV,
                   T*   d_comp)
{
    int tid = threadIdx.x, bid = blockIdx.x;
    int k = ks[bid];
    int r = tid/k, c = tid%k;
    if (r>=k) return;
    int offset_src = offsets_src[bid];
    int offset_dst = offsets_dst[bid];
    if(r == c) {
        d_comp[2*k*(r+k) + c+k + offset_dst] = 1.0;
        d_comp[2*k*r+c+offset_dst] = 1.0;
    } else {
        d_comp[2*k*(r+k) + c+k+offset_dst] = 0.0;
        d_comp[2*k*r+c+offset_dst] = 0.0;
    }
    d_comp[2*k*r+c+k+offset_dst] = dWV[k*(r+k)+c+offset_src];
    d_comp[2*k*(r+k)+c+offset_dst] = dWV[k*r+c+offset_src];
}


template <typename T>
__global__ void
assembleReducedMat_g32(int* ks,
                       int* offsets_src,
                       int* offsets_dst,
                       T*   dWV,
                       T*   d_comp)
{
    int tid = threadIdx.x, bidx = blockIdx.x, bidy = blockIdx.y;
    int k = ks[bidy];
    int r = bidx, c = tid;
    if (r >= k || c >= k) return;
    int offset_src = offsets_src[bidy];
    int offset_dst = offsets_dst[bidy];

    if(r == c) {
        d_comp[2*k*(r+k) + c+k + offset_dst] = 1.0;
        d_comp[2*k*r+c+offset_dst] = 1.0;
    } else {
        d_comp[2*k*(r+k) + c+k+offset_dst] = 0.0;
        d_comp[2*k*r+c+offset_dst] = 0.0;
    }
    d_comp[2*k*r+c+k+offset_dst] = dWV[k*(r+k)+c+offset_src];
    d_comp[2*k*(r+k)+c+offset_dst] = dWV[k*r+c+offset_src];
}


template <typename T>
__global__ void
assembleReducedMat_general(int* ks,
                           int* offsets_src,
                           int* offsets_dst,
                           T*   dWV,
                           T*   d_comp)
{
    int tid = threadIdx.x, bidy = blockIdx.y;
    int k = ks[bidy];
    int r = blockIdx.x, c = tid;
    if (r >= k || c >= k) return;
    int offset_src = offsets_src[bidy];
    int offset_dst = offsets_dst[bidy];

    for(c = tid; c < k; c+=blockDim.x) {
        if(r == c) {
            d_comp[2*k*(r+k) + c+k + offset_dst] = 1.0;
            d_comp[2*k*r+c+offset_dst] = 1.0;
        } else {
            d_comp[2*k*(r+k) + c+k+offset_dst] = 0.0;
            d_comp[2*k*r+c+offset_dst] = 0.0;
        }
        d_comp[2*k*r+c+k+offset_dst] = dWV[k*(r+k)+c+offset_src];
        d_comp[2*k*(r+k)+c+offset_dst] = dWV[k*r+c+offset_src];
    }
}
} // namespace var

template <typename T>
__global__ void matrixVReordering(int  k,
                                  T*   WV,
                                  T*   WV_spare,
                                  int* perms,
                                  int* widths)
{
    int cur_width = widths[blockIdx.y];
    if (blockIdx.x >= cur_width) return;
    int cur_perm = perms[blockIdx.y*k + blockIdx.x];

    for (int tid = threadIdx.x; tid < k; tid+=blockDim.x)
        WV_spare[blockIdx.y*k*k*2 + cur_perm*k + tid] = WV[blockIdx.y*k*k*2 + blockIdx.x*k + tid];
}

template <typename T>
__global__ void matrixWReordering(int  k,
                                  T*   WV,
                                  T*   WV_spare,
                                  int* perms,
                                  int* widths)
{
    int cur_width = widths[blockIdx.y];
    if (blockIdx.x < k-cur_width) return;
    int cur_perm = perms[blockIdx.y*k + blockIdx.x];

    for (int tid = threadIdx.x; tid < k; tid+=blockDim.x)
        WV_spare[(2*blockIdx.y+1)*k*k + cur_perm*k + tid] = WV[(2*blockIdx.y+1)*k*k + blockIdx.x*k + tid];
}

template <typename T>
__global__ void matrixVReordering_perPartition(int  k,
                                               T*   WV,
                                               T*   WV_spare,
                                               int* perms)
{
    int cur_perm = perms[blockIdx.x];

    for (int tid = threadIdx.x; tid < k; tid+=blockDim.x)
        WV_spare[cur_perm*k + tid] = WV[blockIdx.x*k + tid];
}

template <typename T>
__global__ void matrixWReordering_perPartition(int  k,
                                               T*   WV,
                                               T*   WV_spare,
                                               int* perms)
{
    int cur_perm = perms[k-1-blockIdx.x];

    for (int tid = threadIdx.x; tid < k; tid+=blockDim.x)
        WV_spare[cur_perm*k + tid] = WV[(k-1-blockIdx.x)*k + tid];
}

template <typename T>
__global__ void copySeveralColumns(
    T *dA,
    T *dB,
    int a_num_columns,
    int num_columns_to_copy
) {
    int gridY = blockIdx.y, idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_columns_to_copy) {
        return;
    }
    idx += a_num_columns - num_columns_to_copy;
    dA[gridY * a_num_columns + idx] = dB[gridY * a_num_columns + idx];
}

template <typename T>
__global__ void copyBandedMatrixToSegMatrix(
    T *         dst,
    const T *   src,
    int         global_n,
    int         global_num_partitions,
    const int * ks,
    const int * src_offsets,
    const int * dst_offsets_of_offsets,
    const int * dst_offsets,
    bool        upper
) {
    int k                = ks[blockIdx.y];
    int offset           = src_offsets[blockIdx.y];
    int part_size        = global_n / global_num_partitions;
    int global_remainder = global_n % global_num_partitions;

    int local_n          = part_size + (blockIdx.y < global_remainder ? 1 : 0);
    int local_num_partitions = local_n / k;

    if (local_num_partitions == 0) {
        return;
    }

    int local_part_size      = local_n / local_num_partitions;
    int local_remainder      = local_n % local_num_partitions;

    if (upper) {
        for (int i = blockIdx.x; i < local_num_partitions - 1; i += gridDim.x) {
            int dst_offset       = dst_offsets[dst_offsets_of_offsets[blockIdx.y] + i];
            int num_of_rows = local_part_size + (i + 1 < local_remainder ? 1 : 0);
            // int num_of_rows = k;
            int num_of_cols = local_part_size + (i < local_remainder ? 1 : 0);
            // int num_of_cols = k + 1;
            int num_elements = num_of_rows * num_of_cols;
            int src_offset;
            if (i + 1 < local_remainder) {
                src_offset = offset + (2 * k + 1) * ((local_part_size + 1) * (i + 1));
            } else {
                src_offset = offset + (2 * k + 1) * (local_part_size * (i + 1) + local_remainder);
            }

            for (int j = threadIdx.x; j < num_elements; j += blockDim.x) {
                int row_idx = j / num_of_cols;
                int col_idx = j % num_of_cols;

                // src_offset + k + num_of_rows + 2 * k * row_idx + col_idx
                if (k - num_of_cols - row_idx + col_idx >= 0) {
                    dst[dst_offset + col_idx * num_of_rows + row_idx] = src[src_offset + k - num_of_cols + 2 * k * row_idx + col_idx];
                } else {
                    dst[dst_offset + col_idx * num_of_rows + row_idx] = 0;
                }
            }
        }
    } else {
        for (int i = blockIdx.x; i < local_num_partitions - 1; i += gridDim.x) {
            int dst_offset       = dst_offsets[dst_offsets_of_offsets[blockIdx.y] + i];
            int num_of_rows = local_part_size + (i < local_remainder ? 1 : 0);
            // int num_of_rows = k;
            int num_of_cols = local_part_size + (i + 1 < local_remainder ? 1 : 0);
            // int num_of_cols = k + 1;
            int num_elements = num_of_rows * num_of_cols;
            int src_offset;
            if (i < local_remainder) {
                src_offset = offset + (2 * k + 1) * ((local_part_size + 1) * i);
            } else {
                src_offset = offset + (2 * k + 1) * (local_part_size * i + local_remainder);
            }

            for (int j = threadIdx.x; j < num_elements; j += blockDim.x) {
                int row_idx = j / num_of_cols;
                int col_idx = j % num_of_cols;

                // src_offset + k + num_of_rows + 2 * k * row_idx + col_idx
                if (num_of_rows - row_idx + col_idx <= k) {
                    dst[dst_offset + col_idx * num_of_rows + row_idx] = src[src_offset + k + num_of_rows + 2 * k * row_idx + col_idx];
                } else {
                    dst[dst_offset + col_idx * num_of_rows + row_idx] = 0;
                }
            }
        }
    }
}

template <typename T1, typename T2>
__global__ void bandedMatrixTranspose(
    const T1* ori_mat,
    T2*       transposed_mat,
    int       n,
    int       k
) {
    int row = gridDim.x * blockIdx.y + blockIdx.x;

    if (row >= n) {
        return;
    }

    if (transposed_mat != ori_mat) {
        int start_col = row - k, end_col = row + k + 1;
        if (start_col < 0) {
            start_col = 0;
        }

        if (end_col > n) {
            end_col = n;
        }

        for (int i = start_col + threadIdx.x; i < end_col; i += blockDim.x) {
            transposed_mat[2 * k * row + k + i] = ori_mat[2 * k * i + k + row];
        }
    } else {
        int start_col = 1, end_col = k + 1;
        if (row + end_col > n) {
            end_col = n - row;
        }

        for (int i = start_col + threadIdx.x; i < end_col; i += blockDim.x) {
            T1 tmp_value = ori_mat[(2 * k + 1) * row + k + i];
            transposed_mat[(2 * k + 1) * row + k + i] = transposed_mat[(2 * k + 1) * (row + i) + k - i];
            transposed_mat[(2 * k + 1) * (row + i) + k - i] = tmp_value;
        }
    }
}

} //namespace device
} //namespace sap


#endif

