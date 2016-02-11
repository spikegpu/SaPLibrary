#ifndef SAP_SEGMENTED_MATRIX_H
#define SAP_SEGMENTED_MATRIX_H

#include <sap/common.h>
#include <sap/device/data_transfer.cuh>
#include <sap/device/sweep_band_var.cuh>
#include <sap/device/matrix_multiply.cuh>

#include <cusp/array1d.h>
#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#include <cusp/detail/format_utils.h>
#else
#include <cusp/blas/blas.h>
#include <cusp/format_utils.h>
#endif

#include <iostream>
#include <cstdio>

namespace sap {

template <typename Array, typename MemorySpace>
class SegmentedMatrix {
public:
    typedef typename Array::value_type                      T;
    typedef typename Array::value_type                      value_type;
    typedef typename cusp::array1d<T, MemorySpace>          Vector;
    typedef typename cusp::array1d<T, cusp::host_memory>    VectorH;
    typedef typename cusp::array1d<int, MemorySpace>        IntVector;
    typedef typename cusp::array1d<int, cusp::host_memory>  IntVectorH;

    bool        m_upper;
    int         m_global_n;
    int         m_global_num_partitions;

    IntVector   m_ns_scan;
    IntVector   m_ks_per_partition;
    IntVector   m_A_offsets;
    IntVector   m_A_offsets_of_offsets;
    IntVector   m_src_offsets;
    IntVector   m_src_offsets_of_offsets;

    IntVectorH  m_ns_scan_host;
    IntVectorH  m_ks_per_partition_host;
    IntVectorH  m_A_offsets_host;
    IntVectorH  m_A_offsets_of_offsets_host;
    IntVectorH  m_src_offsets_of_offsets_host;
    IntVectorH  m_src_offsets_host;

    Vector      m_A;

    template <typename IntArray>
    void init(
        int               n,
        int               num_partitions,
        const IntArray&   ks_per_parttion,
        bool              upper
    );

    template <typename IntArray>
    void copy(
        const Array&      A,
        const IntArray&   A_offsets
    );

    void sweepStride(
        const Array&      A,
        int   k
    );

    bool multiply_stride(
        SegmentedMatrix<Array, MemorySpace>& result_matrix
    ) const;

    bool multiply_stride(
        const SegmentedMatrix<Array, MemorySpace>& mat_a, 
        const SegmentedMatrix<Array, MemorySpace>& mat_b
    );

    void update_banded(
        Array&  A,
        int     k
    ) const;
};

template <typename Array, typename MemorySpace>
template <typename IntArray>
void
SegmentedMatrix<Array, MemorySpace>::init(
        int               n,
        int               num_partitions,
        const IntArray&   ks_per_partition,
        bool              upper
    )
{
    m_global_n              = n;
    m_global_num_partitions = num_partitions;
    m_ks_per_partition      = ks_per_partition;
    m_ks_per_partition_host = ks_per_partition;
    m_upper                 = upper;

    int global_partition_size = n / num_partitions;
    int global_remainder      = n % num_partitions;

    m_A_offsets_host.push_back(0);
    m_A_offsets_of_offsets_host.push_back(0);

    m_src_offsets_host.push_back(0);
    m_src_offsets_of_offsets_host.push_back(0);

    m_ns_scan_host.push_back(0);

    for (int i = 0; i < num_partitions; i++) {
        int local_n = global_partition_size + (i < global_remainder ? 1 : 0);
        int local_k = m_ks_per_partition_host[i];

        m_ns_scan_host.push_back(m_ns_scan_host.back() + local_n);

        int local_num_partitions = local_n / local_k;

        if (local_num_partitions > 1) {
            int local_part_size = local_n / local_num_partitions;
            int local_remainder = local_n % local_num_partitions;

            IntVectorH tmp_ks(local_num_partitions - 1);

            for (int j = 0; j < local_num_partitions - 1; j++) {
                tmp_ks[j] = (local_part_size + (j + 1 < local_remainder ? 1 : 0)) * (local_part_size + (j < local_remainder ? 1 : 0));
                m_src_offsets_host.push_back(m_src_offsets_host.back() + (local_part_size + (j < local_remainder ? 1 : 0)) * (2 * local_k + 1));
            }
            m_src_offsets_host.push_back(m_src_offsets_host.back() + local_part_size * (2 * local_k + 1));

            tmp_ks[0] += m_A_offsets_host.back();
            for (int j = 0; j < local_num_partitions - 2; j++) {
                tmp_ks[j + 1] += tmp_ks[j];
            }

            m_A_offsets_host.insert(m_A_offsets_host.end(), tmp_ks.begin(), tmp_ks.end());
        }

        m_A_offsets_of_offsets_host.push_back(m_A_offsets_host.size() - 1);
        m_src_offsets_of_offsets_host.push_back(m_src_offsets_host.size() - 1);
    }
    m_A_offsets = m_A_offsets_host;
    m_src_offsets = m_src_offsets_host;
    m_A_offsets_of_offsets = m_A_offsets_of_offsets_host;
    m_src_offsets_of_offsets = m_src_offsets_of_offsets_host;
    m_A.resize(m_A_offsets_host.back(), 0);

    m_ns_scan =  m_ns_scan_host;
}

template <typename Array, typename MemorySpace>
template <typename IntArray>
void
SegmentedMatrix<Array, MemorySpace>::copy(
    const Array&      A,
    const IntArray&   A_offsets
) {
    const T *p_srcA = thrust::raw_pointer_cast(&A[0]);
    T *p_dstA = thrust::raw_pointer_cast(&m_A[0]);
    const int *ks = thrust::raw_pointer_cast(&m_ks_per_partition[0]);
    const int *p_A_offsets = thrust::raw_pointer_cast(&A_offsets[0]);
    const int *p_dst_offsets_of_offsets = thrust::raw_pointer_cast(&m_A_offsets_of_offsets[0]);
    const int *p_dst_offsets = thrust::raw_pointer_cast(&m_A_offsets[0]);

    dim3 grids(m_global_n / m_global_num_partitions / m_ks_per_partition_host[0] + 1,m_global_num_partitions, 1);

    device::copyBandedMatrixToSegMatrix<<<grids, 512>>>(p_dstA, p_srcA, m_global_n, m_global_num_partitions, ks, p_A_offsets, p_dst_offsets_of_offsets, p_dst_offsets, m_upper);
}

template <typename Array, typename MemorySpace>
void
SegmentedMatrix<Array, MemorySpace>::sweepStride(
    const Array&      A,
    int   k
) {
    const int SWEEP_MAX_NUM_THREADS = 128;

    const T *p_srcA = thrust::raw_pointer_cast(&A[0]);
    T *p_dstA = thrust::raw_pointer_cast(&m_A[0]);
    const int *ks = thrust::raw_pointer_cast(&m_ks_per_partition[0]);
    const int *p_dst_offsets_of_offsets = thrust::raw_pointer_cast(&m_A_offsets_of_offsets[0]);
    const int *p_dst_offsets = thrust::raw_pointer_cast(&m_A_offsets[0]);
    const int *p_src_offsets_of_offsets = thrust::raw_pointer_cast(&m_src_offsets_of_offsets[0]);
    const int *p_src_offsets = thrust::raw_pointer_cast(&m_src_offsets[0]);

    const int *p_ns_scan     = thrust::raw_pointer_cast(&m_ns_scan[0]);

    dim3 grids(1, ((m_ns_scan_host[1] - m_ns_scan_host[0]) / m_ks_per_partition_host[0] + 1) / 2 + 1, m_global_num_partitions);
    int gridX = 1;
    int numThreadsPerBlock = k + 1; // FIXME: might be WRONG!!
    kernelConfigAdjust(numThreadsPerBlock, gridX, SWEEP_MAX_NUM_THREADS);
    grids.x = gridX;

    {
        device::var::fwdSweep_stride<<<grids, numThreadsPerBlock>>>(p_srcA, p_dstA, p_ns_scan, ks, p_src_offsets_of_offsets, p_src_offsets, p_dst_offsets_of_offsets, p_dst_offsets, false, m_upper);
    }
    {
        device::var::preBckDivision_stride<<<grids, numThreadsPerBlock>>>(p_srcA, p_dstA, p_ns_scan, ks, p_src_offsets_of_offsets, p_src_offsets, p_dst_offsets_of_offsets, p_dst_offsets, false, m_upper);
    }
    {
        device::var::bckSweep_stride<<<grids, numThreadsPerBlock>>>(p_srcA, p_dstA, p_ns_scan, ks, p_src_offsets_of_offsets, p_src_offsets, p_dst_offsets_of_offsets, p_dst_offsets, false, m_upper);
    }
}

template <typename Array, typename MemorySpace>
bool
SegmentedMatrix<Array, MemorySpace>::multiply_stride(
    SegmentedMatrix<Array, MemorySpace>& result_matrix
) const {
    result_matrix.m_global_num_partitions = m_global_num_partitions;
    result_matrix.m_upper                 = m_upper;
    result_matrix.m_ks_per_partition      = m_ks_per_partition;
    result_matrix.m_ks_per_partition_host = m_ks_per_partition_host;

    int new_n = 0;

    result_matrix.m_ns_scan_host.push_back(0);
    result_matrix.m_A_offsets_of_offsets_host.push_back(0);
    result_matrix.m_A_offsets_host.push_back(0);
    result_matrix.m_src_offsets_of_offsets_host.push_back(0);

    bool non_trivial = false;

    for (int i = 0; i < m_global_num_partitions; i ++) {
        int local_n = m_ns_scan_host[i+1] - m_ns_scan_host[i];
        int local_k = m_ks_per_partition_host[i];

        int local_num_partitions = local_n / local_k;

        if (local_num_partitions > 1) {
            int local_part_size = local_n / local_num_partitions;
            int local_remainder = local_n % local_num_partitions;

            for (int j = 1; j < local_num_partitions; j += 2) {
                new_n += local_part_size + (j < local_remainder ? 1 : 0);
                result_matrix.m_src_offsets_host.push_back(m_src_offsets_host[m_src_offsets_of_offsets_host[i] + j]);

                if (m_upper) {
                    if (j + 2 < local_num_partitions) {
                        result_matrix.m_A_offsets_host.push_back(result_matrix.m_A_offsets_host.back() + (local_part_size + (j < local_remainder ? 1 : 0)) * (local_part_size + (j + 2 < local_remainder ? 1 : 0)));
                        non_trivial = true;
                    }
                } else {
                    if (j > 1) {
                        result_matrix.m_A_offsets_host.push_back(result_matrix.m_A_offsets_host.back() + (local_part_size + (j < local_remainder ? 1 : 0)) * (local_part_size + (j - 2 < local_remainder ? 1 : 0)));
                        non_trivial = true;
                    }
                }
            }
        }

        result_matrix.m_ns_scan_host.push_back(new_n);
        result_matrix.m_src_offsets_of_offsets_host.push_back(result_matrix.m_src_offsets_host.size());
        result_matrix.m_A_offsets_of_offsets_host.push_back(result_matrix.m_A_offsets_host.size() - 1);
    }

    result_matrix.m_ns_scan = result_matrix.m_ns_scan_host;
    result_matrix.m_src_offsets_of_offsets = result_matrix.m_src_offsets_of_offsets_host;
    result_matrix.m_src_offsets = result_matrix.m_src_offsets_host;
    result_matrix.m_A_offsets_of_offsets = result_matrix.m_A_offsets_of_offsets_host;
    result_matrix.m_A_offsets = result_matrix.m_A_offsets_host;
    result_matrix.m_global_n = new_n;

    result_matrix.m_A.resize(result_matrix.m_A_offsets_host.back());

    if (!non_trivial) {
        return false;
    }

    const T*    p_srcA                    = thrust::raw_pointer_cast(&m_A[0]);
    T*          p_dstA                    = thrust::raw_pointer_cast(&result_matrix.m_A[0]);
    const int * p_src_ns_scan             = thrust::raw_pointer_cast(&m_ns_scan[0]);
    const int * p_src_ks                  = thrust::raw_pointer_cast(&m_ks_per_partition[0]);
    const int * p_src_offsets             = thrust::raw_pointer_cast(&m_A_offsets[0]);
    const int * p_src_offsets_of_offsets  = thrust::raw_pointer_cast(&m_A_offsets_of_offsets[0]);
    const int * p_dst_offsets             = thrust::raw_pointer_cast(&result_matrix.m_A_offsets[0]);
    const int * p_dst_offsets_of_offsets  = thrust::raw_pointer_cast(&result_matrix.m_A_offsets_of_offsets[0]);

    int max_k = cusp::blas::nrmmax(m_ks_per_partition_host);
    dim3 blocks(MATRIX_MUL_BLOCK_SIZE, MATRIX_MUL_BLOCK_SIZE, 1);
    dim3 grids(max_k * max_k / MATRIX_MUL_BLOCK_SIZE / MATRIX_MUL_BLOCK_SIZE + 1, ((m_ns_scan_host[1] - m_ns_scan_host[0]) / m_ks_per_partition_host[0] + 1) / 2 + 1, m_global_num_partitions);

    device::negativeMatrixMul<<<grids, blocks>>>(p_srcA, p_src_ns_scan, p_src_ks, p_src_offsets_of_offsets, p_src_offsets, p_dstA, p_dst_offsets_of_offsets, p_dst_offsets, m_upper);

    return true;
}

template <typename Array, typename MemorySpace>
bool
SegmentedMatrix<Array, MemorySpace>::multiply_stride(
    const SegmentedMatrix<Array, MemorySpace>& mat_a,
    const SegmentedMatrix<Array, MemorySpace>& mat_b
) {
    m_global_num_partitions = mat_a.m_global_num_partitions;
    m_upper                 = false;
    m_ks_per_partition      = mat_a.m_ks_per_partition;
    m_ks_per_partition_host = mat_a.m_ks_per_partition_host;

    int new_n = 0;

    m_ns_scan_host.push_back(0);
    m_A_offsets_of_offsets_host.push_back(0);
    m_A_offsets_host.push_back(0);
    m_src_offsets_of_offsets_host.push_back(0);

    bool non_trivial = false;

    for (int i = 0; i < m_global_num_partitions; i ++) {
        int local_n = mat_a.m_ns_scan_host[i+1] - mat_a.m_ns_scan_host[i];
        int local_k = mat_a.m_ks_per_partition_host[i];

        int local_num_partitions = local_n / local_k;

        if (local_num_partitions > 1) {
            int local_part_size = local_n / local_num_partitions;
            int local_remainder = local_n % local_num_partitions;

            for (int j = 1; j < local_num_partitions; j += 2) {
                new_n += local_part_size + (j < local_remainder ? 1 : 0);
                m_src_offsets_host.push_back(mat_a.m_src_offsets_host[mat_a.m_src_offsets_of_offsets_host[i] + j]);

                m_A_offsets_host.push_back(m_A_offsets_host.back() + (local_part_size + (j < local_remainder ? 1 : 0)) * (local_part_size + (j < local_remainder ? 1 : 0)));
                non_trivial = true;

                /*
                if (mat_a.m_upper) {
                    if (j + 1 < local_num_partitions) {
                        m_A_offsets_host.push_back(m_A_offsets_host.back() + (local_part_size + (j < local_remainder ? 1 : 0)) * (local_part_size + (j < local_remainder ? 1 : 0)));
                        non_trivial = true;
                    }
                } else {
                    m_A_offsets_host.push_back(m_A_offsets_host.back() + (local_part_size + (j < local_remainder ? 1 : 0)) * (local_part_size + (j < local_remainder ? 1 : 0)));
                    non_trivial = true;
                }
                */
            }
        }

        m_ns_scan_host.push_back(new_n);
        m_src_offsets_of_offsets_host.push_back(m_src_offsets_host.size());
        m_A_offsets_of_offsets_host.push_back(m_A_offsets_host.size() - 1);
    }

    m_ns_scan = m_ns_scan_host;
    m_src_offsets_of_offsets = m_src_offsets_of_offsets_host;
    m_src_offsets = m_src_offsets_host;
    m_A_offsets_of_offsets = m_A_offsets_of_offsets_host;
    m_A_offsets = m_A_offsets_host;
    m_global_n = new_n;

    m_A.resize(m_A_offsets_host.back());

    if (!non_trivial) {
        return false;
    }

    const T*    p_srcA                    = thrust::raw_pointer_cast(&mat_a.m_A[0]);
    const T*    p_srcB                    = thrust::raw_pointer_cast(&mat_b.m_A[0]);
    T*          p_dstA                    = thrust::raw_pointer_cast(&m_A[0]);
    const int * p_src_ns_scan             = thrust::raw_pointer_cast(&mat_a.m_ns_scan[0]);
    const int * p_src_ks                  = thrust::raw_pointer_cast(&mat_a.m_ks_per_partition[0]);
    const int * p_srcA_offsets             = thrust::raw_pointer_cast(&mat_a.m_A_offsets[0]);
    const int * p_srcA_offsets_of_offsets  = thrust::raw_pointer_cast(&mat_a.m_A_offsets_of_offsets[0]);
    const int * p_srcB_offsets             = thrust::raw_pointer_cast(&mat_b.m_A_offsets[0]);
    const int * p_srcB_offsets_of_offsets  = thrust::raw_pointer_cast(&mat_b.m_A_offsets_of_offsets[0]);
    const int * p_dst_offsets             = thrust::raw_pointer_cast(&m_A_offsets[0]);
    const int * p_dst_offsets_of_offsets  = thrust::raw_pointer_cast(&m_A_offsets_of_offsets[0]);

    int max_k = cusp::blas::nrmmax(m_ks_per_partition_host);
    dim3 blocks(MATRIX_MUL_BLOCK_SIZE, MATRIX_MUL_BLOCK_SIZE, 1);
    dim3 grids(max_k * max_k / MATRIX_MUL_BLOCK_SIZE / MATRIX_MUL_BLOCK_SIZE + 1, ((mat_a.m_ns_scan_host[1] - mat_a.m_ns_scan_host[0]) / mat_a.m_ks_per_partition_host[0] + 1) / 2 + 1, m_global_num_partitions);

    device::negativeMatrixMul<<<grids, blocks>>>(p_srcA, p_srcB, p_src_ns_scan, p_src_ks, p_srcA_offsets_of_offsets, p_srcA_offsets, p_srcB_offsets_of_offsets, p_srcB_offsets, p_dstA, p_dst_offsets_of_offsets, p_dst_offsets, mat_a.m_upper);

    return true;
}

template <typename Array, typename MemorySpace>
void
SegmentedMatrix<Array, MemorySpace>::update_banded(
    Array&  A,
    int     k
) const
{
    const int MAX_NUM_THREADS = 512;

    T *p_srcA = thrust::raw_pointer_cast(&A[0]);
    const T *p_dstA = thrust::raw_pointer_cast(&m_A[0]);
    const int *ks = thrust::raw_pointer_cast(&m_ks_per_partition[0]);
    const int *p_dst_offsets_of_offsets = thrust::raw_pointer_cast(&m_A_offsets_of_offsets[0]);
    const int *p_dst_offsets = thrust::raw_pointer_cast(&m_A_offsets[0]);
    const int *p_src_offsets_of_offsets = thrust::raw_pointer_cast(&m_src_offsets_of_offsets[0]);
    const int *p_src_offsets = thrust::raw_pointer_cast(&m_src_offsets[0]);

    const int *p_ns_scan     = thrust::raw_pointer_cast(&m_ns_scan[0]);

    dim3 grids(k, ((m_ns_scan_host[1] - m_ns_scan_host[0]) / m_ks_per_partition_host[0]) + 1, m_global_num_partitions);
    int numThreadsPerBlock = std::min(k, MAX_NUM_THREADS);
    device::update_banded_matrix<T><<<grids, numThreadsPerBlock>>>(
        p_srcA,
        p_dstA,
        p_ns_scan,
        ks,
        p_src_offsets_of_offsets,
        p_src_offsets,
        p_dst_offsets_of_offsets,
        p_dst_offsets
    ); 
}

}

#endif // SAP_SEGMENTED_MATRIX_H
