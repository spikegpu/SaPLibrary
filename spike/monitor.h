#ifndef SPIKE_MONITOR_H
#define SPIKE_MONITOR_H

#include <limits>
#include <string>

#include <cusp/array1d.h>
#include <cusp/blas/blas.h>


namespace spike {


// ----------------------------------------------------------------------------
// Monitor
//
// This class provides support for monitoring progress of iterative linear
// solvers, check for convergence, and stop on various error conditions.
// ----------------------------------------------------------------------------
template <typename SolverVector>
class Monitor
{
public:
	typedef typename SolverVector::value_type  SolverValueType;

	Monitor(const int              maxIterations,
	        const SolverValueType  relTol,
	        const SolverValueType  absTol = SolverValueType(0));
	~Monitor() {}

	// Initialize the monitor with the specified rhs vector.
	void init(const SolverVector& rhs);

	// Check if we are done (success or failure).
	// Return true if:
	//   (1) stop() was invoked to set code != 0
	//   (2) the given residual has norm below the tolerance (code = 1)
	//   (3) the given residual has norm NaN (code = -2)
	//   (4) the iteration limit was reached (code = -1)
	// Otherwise, return false (code = 0) to continue iterations.
	bool finished(const SolverVector& r);
	bool finished(SolverValueType rNorm);

	// Force stop, specifying the reason (as a code and message). 
	// A positive code indicates success; a negative code indicates failure.
	// If invoked by the solver, a success code should be >= 10 and a failure code <= -10.
	void stop(int code, const char* message)
	{
		m_code = code;
		m_message = message;
	}

	// Increment the iteration count by the specified value.
	void increment(float incr) {m_iterations += incr;}

	// Prefix increment: increase iteration count by 1.
	Monitor& operator++()    {m_iterations += 1.f; return *this;}

	int                getMaxIterations() const   {return m_maxIterations;}
	size_t             iteration_limit()  const   {return (size_t)(m_maxIterations);}
	SolverValueType    getRelTolerance() const    {return m_relTol;}
	SolverValueType    getAbsTolerance() const    {return m_absTol;}
	SolverValueType    getTolerance() const       {return m_absTol + m_relTol * m_rhsNorm;}
	SolverValueType    getRHSNorm() const         {return m_rhsNorm;}

	bool               converged() const          {return m_code > 0;}
	size_t             iteration_count()  const   {return (size_t)(m_iterations + 0.5f);}
	float              getNumIterations() const   {return m_iterations;}
	int                getCode() const            {return m_code;}
	const std::string& getMessage() const         {return m_message;}
	SolverValueType    getResidualNorm() const    {return m_rNorm;}
	SolverValueType    getRelResidualNorm() const {return m_rNorm / m_rhsNorm;}

private:
	int              m_maxIterations;
	float            m_iterations;

	SolverValueType  m_relTol;
	SolverValueType  m_absTol;
	SolverValueType  m_rhsNorm;
	SolverValueType  m_rNorm;

	int              m_code;
	std::string      m_message;
};


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
template <typename SolverVector>
inline
Monitor<SolverVector>::Monitor(const int              maxIterations,
                               const SolverValueType  relTol,
                               const SolverValueType  absTol)
:	m_rNorm(std::numeric_limits<SolverValueType>::max()),
	m_maxIterations(maxIterations),
	m_relTol(relTol),
	m_absTol(absTol),
	m_iterations(0),
	m_code(0),
	m_message("")
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
	m_code = 0;
	m_message = "";
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

	if (m_code != 0)
		return true;

	if (isnan(m_rNorm))                      stop(-2, "Residual norm is NaN");
	else if (m_rNorm <= getTolerance())      stop( 1, "Converged");
	else if (m_iterations > m_maxIterations) stop(-1, "Maximum number of iterations was reached");

	return m_code != 0;
}


} // namespace spike


#endif
