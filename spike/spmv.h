/** \file spmv.h
 *  \brief Definition of the default Spike SPMV functor class.
 */

#ifndef SPIKE_SPMV_H
#define SPIKE_SPMV_H


#include <cusp/multiply.h>

#include <spike/timer.h>


namespace spike {


/// Default SPMV functor class.
/**
 * This class implements the default SPMV functor for sparse matrix-vector
 * product, using the cusp::multiply algorithm.
 *
 * \tparam Matrix is the type of the sparse matrix.
 */
template <typename Matrix>
class SpmvCusp {
public:
	SpmvCusp(Matrix& A) : m_A(A), m_time(0), m_count(0) {}

	/// Cummulative time for all SPMV calls (ms).
	double getTime() const   {return m_time;}

	/// Total number of calls to the SPMV functor.
	double getCount() const  {return m_count;}

	/// Average GFLOP/s over all SPMV calls.
	double getGFlops() const
	{
		double avgTime = m_time / m_count;
		return 2 * m_A.num_entries / (1e6 * avgTime);
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
	Matrix&      m_A;

private:
	GPUTimer     m_timer;
	double       m_time;
	int          m_count;
};

} // namespace spike


#endif
