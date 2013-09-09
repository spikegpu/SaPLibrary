#ifndef SPIKE_SPMV_H
#define SPIKE_SPMV_H


#include <cusp/blas.h>

#include <spike/timer.h>


namespace spike {


// ----------------------------------------------------------------------------
// SpmvCusp
//
// This is the default CUSP-based SPMV functor class for the SPIKE solver.
// It uses the cusp sparse matrix-vector multiply algorithm.
// ----------------------------------------------------------------------------
template <typename Matrix, typename Vector>
class SpmvCusp {
public:
	SpmvCusp(Matrix& A)
		: m_A(A), m_time(0), m_count(0) {}

	double getTime() const   {return m_time;}     // Total time (ms)
	double getCount() const  {return m_count;}    // Total number of calls
	double getGFlops() const {                    // Average GFLOP/s
		double avgTime = m_time / m_count;
		int    nnz = m_A.num_entries;

		return 2 * nnz / (1e6 * avgTime);
	}

	void operator()(const Vector& v,
	                Vector&       Av)
	{
		m_timer.Start();
		cusp::multiply(m_A, v, Av);
		m_timer.Stop();

		m_count++;
		m_time += m_timer.getElapsed();
	}

private:
	Matrix&      m_A;
	GPUTimer     m_timer;
	double       m_time;
	int          m_count;
};

} // namespace spike


#endif
