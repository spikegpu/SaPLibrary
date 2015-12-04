/** \file matrix_multiply.cuh
 *
 * CUDA kernels for segmented matrix multiplication.
 */

#ifndef MATRIX_MULTIPLY_CUH
#define MATRIX_MULTIPLY_CUH

namespace sap {
namespace device {

template <typename T>
__device__ void
negativeMatrixMulAux(
    const T*  dA,
    const T*  dB,
    T*        dC,
    int       A_rows,
    int       A_cols,
    int       B_cols,
    T         A_shared[][MATRIX_MUL_BLOCK_SIZE],
    T         B_shared[][MATRIX_MUL_BLOCK_SIZE]
) {
    int num_block_rows = (A_rows + MATRIX_MUL_BLOCK_SIZE - 1) / MATRIX_MUL_BLOCK_SIZE;
    int num_block_cols = (B_cols + MATRIX_MUL_BLOCK_SIZE - 1) / MATRIX_MUL_BLOCK_SIZE;
    int num_blocks = num_block_rows * num_block_cols;


    int num_A_block_cols = (A_cols + MATRIX_MUL_BLOCK_SIZE - 1) / MATRIX_MUL_BLOCK_SIZE;

    for (int bidx = blockIdx.x; bidx < num_blocks; bidx += gridDim.x) {
        int block_row_idx = bidx / num_block_cols;
        int block_col_idx = bidx % num_block_cols;

        T sum = T(0);

        for (int i = 0; i < num_A_block_cols; i++) {
            if (block_row_idx * MATRIX_MUL_BLOCK_SIZE + threadIdx.y >= A_rows || i * MATRIX_MUL_BLOCK_SIZE + threadIdx.x >= A_cols) {
                A_shared[threadIdx.y][threadIdx.x] = T(0);
            } else {
                A_shared[threadIdx.y][threadIdx.x] = dA[(block_row_idx * MATRIX_MUL_BLOCK_SIZE + threadIdx.y) * A_cols + (i * MATRIX_MUL_BLOCK_SIZE + threadIdx.x)];
            }

            if (i * MATRIX_MUL_BLOCK_SIZE + threadIdx.y >= A_cols || block_col_idx * MATRIX_MUL_BLOCK_SIZE + threadIdx.x >= B_cols) {
                B_shared[threadIdx.y][threadIdx.x] = T(0);
            } else {
                B_shared[threadIdx.y][threadIdx.x] = dB[(i * MATRIX_MUL_BLOCK_SIZE + threadIdx.y) * B_cols + (block_col_idx * MATRIX_MUL_BLOCK_SIZE + threadIdx.x)];
            }

            __syncthreads();

            for (int j = 0; j < MATRIX_MUL_BLOCK_SIZE; j++) {
                sum -= A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
            }

            __syncthreads();
        }

        if (block_row_idx * MATRIX_MUL_BLOCK_SIZE + threadIdx.y < A_rows && block_col_idx * MATRIX_MUL_BLOCK_SIZE + threadIdx.x < B_cols) {
            dC[(block_row_idx * MATRIX_MUL_BLOCK_SIZE + threadIdx.y) * B_cols + (block_col_idx * MATRIX_MUL_BLOCK_SIZE + threadIdx.x)] = sum;
        }

    }
}

template <typename T>
__global__ void
negativeMatrixMul(
    const T*       src_mat,
    const int*     src_ns_scan,
    const int*     src_ks,
    const int*     src_offsets_of_offsets,
    const int*     src_offsets,
    T*             dst_mat,
    const int*     dst_offsets_of_offsets,
    const int*     dst_offsets,
    bool           upper
) {
    __shared__ T A_shared[MATRIX_MUL_BLOCK_SIZE][MATRIX_MUL_BLOCK_SIZE];
    __shared__ T B_shared[MATRIX_MUL_BLOCK_SIZE][MATRIX_MUL_BLOCK_SIZE];

    int n = src_ns_scan[blockIdx.z + 1] - src_ns_scan[blockIdx.z];
    int k = src_ks[blockIdx.z];

    int local_num_partitions = n / k;

    if (local_num_partitions <= 2) {
        return;
    }

    int local_part_size      = n / local_num_partitions;
    int local_remainder      = n % local_num_partitions;

    int it_last = local_num_partitions - (upper ? 3 : 2);

    for (int i = (upper ? 0 : 1) + (blockIdx.y << 1); i < it_last; i += (gridDim.y << 1) ) {
        const T*  A = src_mat + src_offsets[src_offsets_of_offsets[blockIdx.z] + (i + 1)];
        const T*  B = src_mat + src_offsets[src_offsets_of_offsets[blockIdx.z] + i + (upper ? 2 : 0)];
        T*        C = dst_mat + dst_offsets[dst_offsets_of_offsets[blockIdx.z] + (i >> 1)];

        int A_rows;
        int A_cols;
        int B_cols;

        if (upper) {
            A_rows = local_part_size + (i + 1 < local_remainder ? 1 : 0);
            A_cols = local_part_size + (i + 2 < local_remainder ? 1 : 0);
            B_cols = local_part_size + (i + 3 < local_remainder ? 1 : 0);
        } else {
            A_rows = local_part_size + (i + 2 < local_remainder ? 1 : 0);
            A_cols = local_part_size + (i + 1 < local_remainder ? 1 : 0);
            B_cols = local_part_size + (i < local_remainder ? 1 : 0);
        }

        negativeMatrixMulAux(A, B, C, A_rows, A_cols, B_cols, A_shared, B_shared);
    }
}

template <typename T>
__device__ void
matrixVecMulAux(
    const T*  mat,
    const T*  vec1,
    T*        vec2,
    int       mat_rows,
    int       mat_cols,
    T*        shared
) {
    int num_blocks = (mat_cols + MAT_VEC_MUL_BLOCK_SIZE - 1) / MAT_VEC_MUL_BLOCK_SIZE;
    for (int i = blockIdx.x; i < mat_rows; i += gridDim.x) {
        T my_sum = T(0);
        for (int j = 0; j < num_blocks; j++) {
            int offset = MAT_VEC_MUL_BLOCK_SIZE * j + threadIdx.x;
            shared[threadIdx.x] = ((offset < mat_cols) ? (mat[mat_cols * i + offset] * vec1[offset]) : T(0));

            __syncthreads();

            for (int k = 64; k > 1; k >>= 1) {
                if (threadIdx.x < k) {
                    shared[threadIdx.x] += shared[threadIdx.x + k];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                my_sum += shared[0] + shared[1];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            vec2[i] -= my_sum;
        }
    }
}

template <typename T>
__global__ void
matrixVecMul(
    const T*      mat,
    const T*      vec1,
    T*            vec2,
    int           n,
    int           num_partitions,
    const int*    ns_scan,
    const int*    ks,
    const int*    offsets_of_offsets,
    const int*    offsets,
    int           stride,
    bool          backward,
    bool          upper
) {
    __shared__ T shared[MAT_VEC_MUL_BLOCK_SIZE];

    int k      = ks[blockIdx.z];
    int part_size = n / num_partitions;
    int remainder = n % num_partitions;
    int local_n   = part_size + (blockIdx.z < remainder ? 1 : 0);

    int local_num_partitions = local_n / k;
    int num_block_matrices   = (ns_scan[blockIdx.z + 1] - ns_scan[blockIdx.z]) / k - 1;
    if (num_block_matrices < 1) {
        return;
    }

    int local_part_size      = local_n / local_num_partitions;
    int local_remainder      = local_n % local_num_partitions;
    int vec_offset_base;

    if (blockIdx.z < remainder) {
        vec_offset_base = blockIdx.z * (part_size + 1);
    } else {
        vec_offset_base = blockIdx.z * part_size + remainder;
    }

    for (int i = (blockIdx.y << 1) + ((upper ^ backward) ? 1 : 0); i < num_block_matrices; i += (gridDim.y << 1)) {
        int offset         = offsets[offsets_of_offsets[blockIdx.z] + i];
        int vec1_block_idx = (i >> 1) * stride + stride / 2 + (upper ? stride : 0)- 1;
        int vec2_block_idx = vec1_block_idx + (upper ? (-stride / 2) : (stride / 2));

        if (backward) {
            vec1_block_idx = vec2_block_idx;
            vec2_block_idx = vec1_block_idx + (upper ? (-stride / 2) : (stride / 2));
        }

        if (vec1_block_idx >= local_num_partitions || vec2_block_idx >= local_num_partitions) {
            continue;
        }

        int vec1_offset, vec2_offset;

        int mat_rows, mat_cols;

        if (vec1_block_idx < local_remainder) {
            vec1_offset = vec1_block_idx * (local_part_size + 1);
            mat_cols = (local_part_size + 1);
        } else {
            vec1_offset = vec1_block_idx * local_part_size + local_remainder;
            mat_cols = local_part_size;
        }

        if (vec2_block_idx < local_remainder) {
            vec2_offset = vec2_block_idx * (local_part_size + 1);
            mat_rows = (local_part_size + 1);
        } else {
            vec2_offset = vec2_block_idx * local_part_size + local_remainder;
            mat_rows = local_part_size;
        }

        matrixVecMulAux(
            mat + offset,
            vec1 + vec_offset_base + vec1_offset,
            vec2 + vec_offset_base + vec2_offset,
            mat_rows,
            mat_cols,
            shared
        );
    }
}

} // namespace device
} // namespace sap

#endif // MATRIX_MULTIPLY_CUH
