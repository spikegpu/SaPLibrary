/** \file spmv.h
 *  \brief Definition of the default Spike SPMV functor class.
 */

#ifndef SPIKE_SPMV_H
#define SPIKE_SPMV_H


#include <cusp/multiply.h>

#include <spike/timer.h>

#include "cusparse.h"


namespace spike {


/// Default SPMV functor class.
/**
 * This class implements the default SPMV functor for sparse matrix-vector
 * product, using the cusp::multiply algorithm.
 *
 * \tparam Matrix is the type of the sparse matrix.
 */
template <typename Matrix>
class SpmvCusp : public cusp::linear_operator<typename Matrix::value_type, typename Matrix::memory_space, typename Matrix::index_type> 
{
public:
	typedef typename cusp::linear_operator<typename Matrix::value_type, typename Matrix::memory_space, typename Matrix::index_type> Parent;

	SpmvCusp(Matrix& A) 
	:	Parent(A.num_rows, A.num_cols),
		m_A(A),
		m_time(0), 
		m_count(0) 
	{}

	/// Cummulative time for all SPMV calls (ms).
	double getTime() const   {return m_time;}

	/// Total number of calls to the SPMV functor.
	double getCount() const  {return m_count;}

	/// Average GFLOP/s over all SPMV calls.
	double getGFlops() const
	{
		// double avgTime = m_time / m_count;
		// return 2 * m_A.num_entries / (1e6 * avgTime);
		return 0;
	}

	/// Implementation of the SPMV functor using cusp::multiply().
	template <typename Array>
	void operator()(const Array& v,
	                Array&       Av)
	{
		// m_timer.Start();
		cusp::multiply(m_A, v, Av);
		// m_timer.Stop();

		// m_count++;
		// m_time += m_timer.getElapsed();
	}

private:
	Matrix&      m_A;

	GPUTimer     m_timer;
	double       m_time;
	int          m_count;
};

/**
 * This class implements the default SPMV functor for sparse matrix-vector
 * product, using cuSparse algorithm.
 *
 * \tparam Matrix is the type of the sparse matrix.
 */
template <typename Matrix>
class SpmvCuSparse : public cusp::linear_operator<typename Matrix::value_type, typename Matrix::memory_space, typename Matrix::index_type> 
{
public:
	typedef typename cusp::linear_operator<typename Matrix::value_type, typename Matrix::memory_space, typename Matrix::index_type> Parent;

	typedef typename Matrix::value_type fp_type;
	typedef typename Matrix::index_type idx_type;

	SpmvCuSparse(Matrix& A, cusparseHandle_t& handle) 
	:	Parent(A.num_rows, A.num_cols),
		m_A(A),
		m_time(0), 
		m_count(0),
		m_handle(handle)
	{
		cusparseCreateMatDescr(&m_descr);
		cusparseSetMatType(m_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatDiagType(m_descr,CUSPARSE_DIAG_TYPE_NON_UNIT);
		cusparseSetMatIndexBase(m_descr,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatFillMode(m_descr, CUSPARSE_FILL_MODE_LOWER);

		cusparseCreateHybMat(&m_hybA);

		typename Matrix::format format1;
		fp_type  *p_tmp = NULL;
		toHybMat(format1, p_tmp);
	}

	~SpmvCuSparse() {
		cusparseDestroyMatDescr(m_descr);
		cusparseDestroyHybMat(m_hybA);
	}

	/// Cummulative time for all SPMV calls (ms).
	double getTime() const   {return m_time;}

	/// Total number of calls to the SPMV functor.
	double getCount() const  {return m_count;}

	/// Average GFLOP/s over all SPMV calls.
	double getGFlops() const
	{
		// double avgTime = m_time / m_count;
		// return 2 * m_A.num_entries / (1e6 * avgTime);
		return 0;
	}

	/// Implementation of the SPMV functor using cusp::multiply().
	template <typename Array>
	void operator()(const Array& v,
	                Array&       Av)
	{
		// m_timer.Start();
		typename Matrix::format format1;
		spmv(v, Av, format1);
		// m_timer.Stop();

		// m_count++;
		// m_time += m_timer.getElapsed();
	}

private:
	Matrix&            m_A;
	cusparseHybMat_t   m_hybA;

	GPUTimer           m_timer;
	double             m_time;
	int                m_count;
	cusparseHandle_t&  m_handle;
	cusparseMatDescr_t m_descr;

	void toHybMat(typename   cusp::csr_format, double *) {
		cusparseDcsr2hyb(m_handle, m_A.num_rows, m_A.num_rows, m_descr, thrust::raw_pointer_cast(&m_A.values[0]), thrust::raw_pointer_cast(&m_A.row_offsets[0]), thrust::raw_pointer_cast(&m_A.column_indices[0]), m_hybA, m_A.num_entries, CUSPARSE_HYB_PARTITION_AUTO);
	}

	void toHybMat(typename   cusp::csr_format, float *) {
		cusparseScsr2hyb(m_handle, m_A.num_rows, m_A.num_rows, m_descr, thrust::raw_pointer_cast(&m_A.values[0]), thrust::raw_pointer_cast(&m_A.row_offsets[0]), thrust::raw_pointer_cast(&m_A.column_indices[0]), m_hybA, m_A.num_entries, CUSPARSE_HYB_PARTITION_AUTO);
	}

	void toHybMat(typename   cusp::coo_format, fp_type *) {}

	template <typename Array>
	void spmv(const Array& v,
	          Array&       Av,
			  typename     cusp::csr_format)
	{
		typename cusp::csr_format format1;
		const fp_type *p_v = thrust::raw_pointer_cast(&v[0]);
		fp_type *p_Av      = thrust::raw_pointer_cast(&Av[0]);
		idx_type *row_offsets    = thrust::raw_pointer_cast(&m_A.row_offsets[0]);
		idx_type *column_indices = thrust::raw_pointer_cast(&m_A.column_indices[0]);
		fp_type  *values         = thrust::raw_pointer_cast(&m_A.values[0]);
		spmv_inner(row_offsets, column_indices, values, p_v, p_Av, format1);
	}

	template <typename Array>
	void spmv(const Array& v,
	          Array&       Av,
			  typename     cusp::coo_format)
	{
		cusp::multiply(m_A, v, Av);
	}

	void spmv_inner(idx_type     *row_offsets,
					idx_type     *column_indices,
					double       *values,
					const double *p_v,
					double       *p_Av,
					typename     cusp::csr_format) {
		double one  = 1.0;
		double zero = 0.0;
		//// cusparseDcsrmv(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_A.num_rows, m_A.num_cols, m_A.num_entries, &one, m_descr, values, row_offsets, column_indices, p_v, &zero, p_Av);
		cusparseDhybmv(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, m_descr, m_hybA, p_v, &zero, p_Av);
	}

	void spmv_inner(idx_type     *row_offsets,
					idx_type     *column_indices,
					float        *values,
					const float  *p_v,
					float        *p_Av,
					typename     cusp::csr_format) {
		float one  = 1.f;
		float zero = 0.f;
		//// cusparseScsrmv(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m_A.num_rows, m_A.num_cols, m_A.num_entries, &one, m_descr, values, row_offsets, column_indices, p_v, &zero, p_Av);
		cusparseShybmv(m_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, m_descr, m_hybA, p_v, &zero, p_Av);
	}
};

} // namespace spike


#endif
