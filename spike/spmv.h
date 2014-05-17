/** \file spmv.h
 *  \brief Definition of the default Spike SPMV functor class.
 */

#ifndef SPIKE_SPMV_H
#define SPIKE_SPMV_H


#include <cusparse.h>

#include <spike/timer.h>


namespace spike {


/// Default SPMV functor class.
/**
 * This class implements the default SPMV functor for sparse matrix-vector
 * product, using the cusparse multiplication algorithm csrmv.
 *
 * \tparam T is the type of the sparse matrix values.
 */
template <typename T>
class Spmv
{
public:
	Spmv(CSRMatrix<T>& A) 
	:	m_A(A),
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

	/// Implementation of the SPMV functor
	void operator()(const Vector<T>& v,
	                Vector<T>&       Av)
	{
		// m_timer.Start();
		
		//// TODO:  this needs to be fixed !!!!
		////cusparseDcsrmv(...);

		// m_timer.Stop();

		// m_count++;
		// m_time += m_timer.getElapsed();
	}

private:
	CSRMatrix<T>&   m_A;

	GPUTimer        m_timer;
	double          m_time;
	int             m_count;
};

} // namespace spike


#endif
