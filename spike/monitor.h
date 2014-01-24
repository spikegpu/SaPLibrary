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
		FailedLimit,
		FailedNaN,
		FailedOther
	};

	Monitor(const int              maxIterations,
	        const SolverValueType  tolerance);
	~Monitor() {}

	// Initialize the monitor with the specified rhs vector.
	void init(const SolverVector& rhs);

	// Check if we are done. Return true if:
	//   (1) the given residual has relative norm below the tolerance (state = Converged)
	//   (2) the given residual has norm NaN (state = FailedNaN)
	//   (3) the iteration limit was reached (state = FailedLimit)
	//   (4) the monitor state was forced to FailedOther
	// Otherwise, return false (state = Continue)
	bool finished(const SolverVector& r);
	bool finished(SolverValueType rNorm);

	// Force the monitor state to FailOther and cache the error code.
	// The next call to finished() will report true.
	void fail(int errCode) {m_errCode = errCode; m_state = FailedOther;}

	// Increment the iteration count by the specified value.
	void increment(float incr) {m_iterations += incr;}

	// Prefix increment: increase iteration count by 1.
	Monitor& operator++()    {m_iterations += 1.f; return *this;}

	bool             converged() const          {return m_state == Converged;}
	int              getMaxIterations() const   {return m_maxIterations;}
	size_t           iteration_count()  const   {return (size_t)(m_iterations + 0.5f);}
	size_t           iteration_limit()  const   {return (size_t)(m_maxIterations);}
	float            getNumIterations() const   {return m_iterations;}
	int              getErrorCode() const       {return m_errCode;}
	State            getState() const           {return m_state;}
	SolverValueType  getTolerance() const       {return m_tolerance;}
	SolverValueType  getRHSNorm() const         {return m_rhsNorm;}
	SolverValueType  getResidualNorm() const    {return m_rNorm;}
	SolverValueType  getRelResidualNorm() const {return m_rNorm / m_rhsNorm;}

private:
	State            m_state;
	int              m_maxIterations;
	float            m_iterations;
	int              m_errCode;

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
	m_errCode(0),
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
	m_errCode = 0;
	m_state = Continue;
}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
template <typename SolverVector>
inline bool
Monitor<SolverVector>::finished(const SolverVector& r)
{
	return finished(cusp::blas::nrm2(r));
}

template <typename SolverVector>
inline bool
Monitor<SolverVector>::finished(SolverValueType rNorm)
{
	m_rNorm = rNorm;

	if (m_rNorm <= m_tolerance * m_rhsNorm)  m_state = Converged;
	else if (isnan(m_rNorm))                 m_state = FailedNaN;
	else if (m_iterations > m_maxIterations) m_state = FailedLimit;

	return m_state != Continue;
}


} // namespace spike


#endif
