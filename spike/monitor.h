#ifndef SPIKE_MONITOR_H
#define SPIKE_MONITOR_H

#include <limits>

#include <cusp/array1d.h>
#include <cusp/blas.h>


namespace spike {


// ----------------------------------------------------------------------------
// Monitor
//
// This class encapsulates...
// ----------------------------------------------------------------------------
template <typename SolverVector>
class Monitor
{
public:
	typedef typename SolverVector::value_type  SolverValueType;

	enum State {
		Continue,
		Converged,
		Failed
	};

	Monitor(const int              maxIterations,
	        const SolverValueType  tolerance);
	~Monitor() {}

	void init(const SolverVector& rhs);

	bool finished(const SolverVector& r);

	void increment(float incr) {m_iterations += incr;}

	Monitor& operator++(int) {m_iterations += 1.f; return *this;}
	Monitor& operator++()    {m_iterations += 1.f; return *this;}

	bool             converged() const          {return m_state == Converged;}
	int              getMaxIterations() const   {return m_maxIterations;}
	size_t           iteration_count()  const   {return (size_t)(m_iterations + 0.5f);}
	size_t           iteration_limit()  const   {return (size_t)(m_maxIterations);}
	float            getNumIterations() const   {return m_iterations;}
	SolverValueType  getTolerance() const       {return m_tolerance;}
	SolverValueType  getRHSNorm() const         {return m_rhsNorm;}
	SolverValueType  getResidualNorm() const    {return m_rNorm;}
	SolverValueType  getRelResidualNorm() const {return m_rNorm / m_rhsNorm;}

private:
	State            m_state;
	int              m_maxIterations;
	float            m_iterations;
	
	SolverValueType  m_tolerance;
	SolverValueType  m_rhsNorm;
	SolverValueType  m_rNorm;
};


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
template <typename SolverVector>
inline
Monitor<SolverVector>::Monitor(const int              maxIterations,
                               const SolverValueType  tolerance)
:	m_rNorm(std::numeric_limits<SolverValueType>::max()),
	m_maxIterations(maxIterations),
	m_tolerance(tolerance),
	m_iterations(0),
	m_state(Continue)
{
}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
template <typename SolverVector>
inline void
Monitor<SolverVector>::init(const SolverVector& rhs)
{
	m_rhsNorm = cusp::blas::nrm2(rhs);
	m_iterations = 0;
}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
template <typename SolverVector>
inline bool
Monitor<SolverVector>::finished(const SolverVector& r)
{
	m_rNorm = cusp::blas::nrm2(r);

	if (m_rNorm <= m_tolerance * m_rhsNorm)  m_state = Converged;
	else if (isnan(m_rNorm))                 m_state = Failed;
	else if (m_iterations > m_maxIterations) m_state = Failed;
	else                                     m_state = Continue;

	return m_state != Continue;
}


} // namespace spike


#endif
