/** \file monitor.h
 *  \brief It provides two classes for monitoring progress of iterative linear solvers, checking for convergence, and stopping on various error conditions.
 */

#ifndef SAP_MONITOR_H
#define SAP_MONITOR_H

#include <limits>
#include <string>

#include <cusp/array1d.h>
#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif


namespace sap {


// Monitor
/** This class provides support for monitoring progress of iterative linear solvers, check for convergence, and stop on various error conditions.
 */
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
	virtual void init(const SolverVector& rhs);

	// Check if we are done (success or failure).
	// Return true if:
	//   (1) stop() was invoked to set code != 0
	//   (2) the given residual has norm below the tolerance (code = 1)
	//   (3) the given residual has norm NaN (code = -2)
	//   (4) the iteration limit was reached (code = -1)
	// Otherwise, return false (code = 0) to continue iterations.
	virtual bool finished(const SolverVector& r);
	virtual bool finished(SolverValueType rNorm);

	// Force stop, specifying the reason (as a code and message). 
	// A positive code indicates success; a negative code indicates failure.
	// If invoked by the solver, a success code should be >= 10 and a failure code <= -10.
	virtual void stop(int code, const char* message)
	{
		m_code = code;
		m_message = message;
	}

	// Increment the iteration count by the specified value.
	virtual void increment(float incr) {m_iterations += incr;}

	// Prefix increment: increase iteration count by 1.
	virtual Monitor<SolverVector>& operator++()    {m_iterations += 1.f; return *this;}

	virtual int                getMaxIterations() const   {return m_maxIterations;}
	virtual size_t             iteration_limit()  const   {return (size_t)(m_maxIterations);}
	virtual SolverValueType    getRelTolerance() const    {return m_relTol;}
	virtual SolverValueType    getAbsTolerance() const    {return m_absTol;}
	virtual SolverValueType    getTolerance() const       {return m_absTol + m_relTol * m_rhsNorm;}
	virtual SolverValueType    getRHSNorm() const         {return m_rhsNorm;}

	virtual bool               converged() const          {return m_code > 0;}
	virtual size_t             iteration_count()  const   {return (size_t)(m_iterations + 0.5f);}
	virtual float              getNumIterations() const   {return m_iterations;}
	virtual int                getCode() const            {return m_code;}
	virtual const std::string& getMessage() const         {return m_message;}
	virtual SolverValueType    getResidualNorm() const    {return m_rNorm;}
	virtual SolverValueType    getRelResidualNorm() const {return m_rNorm / m_rhsNorm;}

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

// BiCGStabLMonitor
/**
 * This class provides support for monitoring progress of BiCGStab(L) solvers, check for convergence, and stop on various error conditions.
 */
template <typename SolverVector>
class BiCGStabLMonitor
{
public:
	typedef typename SolverVector::value_type  SolverValueType;

	BiCGStabLMonitor(const int              maxIterations,
                     const int              maxmSteps,
	                 const SolverValueType  relTol,
	                 const SolverValueType  absTol = SolverValueType(0));
	~BiCGStabLMonitor() {}

	// Initialize the monitor with the specified rhs vector.
	virtual void init(const SolverVector& rhs);

	// Check if we are done (success or failure).
	// Return true if:
	//   (1) stop() was invoked to set code != 0
	//   (2) the given residual has norm below the tolerance (code = 1)
	//   (3) the given residual has norm NaN (code = -2)
	//   (4) the iteration limit was reached (code = -1)
	// Otherwise, return false (code = 0) to continue iterations.
	virtual bool finished(const SolverVector& r);
	virtual bool finished(SolverValueType rNorm);
	virtual bool finished() {
        if (m_code != 0) {
            return true;
        }
        
        if (m_iterations > m_maxIterations) stop(-1, "Maximum number of iterations was reached");
        return m_code != 0;
    }

	virtual bool needCheckConvergence(const SolverVector& r);
    virtual bool needCheckConvergence(SolverValueType rNorm);

	// Force stop, specifying the reason (as a code and message). 
	// A positive code indicates success; a negative code indicates failure.
	// If invoked by the solver, a success code should be >= 10 and a failure code <= -10.
	virtual void stop(int code, const char* message)
	{
		m_code = code;
		m_message = message;
	}

	// Increment the iteration count by the specified value.
	virtual void increment(float incr) {m_iterations += incr;}

	virtual void incrementStag()       {m_stag ++;}
	virtual void resetStag()           {m_stag = 0;}

    virtual void updateResidual(SolverValueType r)        {m_rNorm = r;}

	// Prefix increment: increase iteration count by 1.
	virtual BiCGStabLMonitor<SolverVector>& operator++()    {m_iterations += 1.f; return *this;}

	virtual int                getMaxIterations() const   {return m_maxIterations;}
	virtual size_t             iteration_limit()  const   {return (size_t)(m_maxIterations);}
	virtual SolverValueType    getRelTolerance() const    {return m_relTol;}
	virtual SolverValueType    getAbsTolerance() const    {return m_absTol;}
	virtual SolverValueType    getTolerance() const       {return m_absTol + m_relTol * m_rhsNorm;}
	virtual SolverValueType    getRHSNorm() const         {return m_rhsNorm;}

	virtual bool               converged() const          {return m_code > 0;}
	virtual size_t             iteration_count()  const   {return (size_t)(m_iterations + 0.5f);}
	virtual float              getNumIterations() const   {return m_iterations;}
	virtual int                getCode() const            {return m_code;}
	virtual const std::string& getMessage() const         {return m_message;}
	virtual SolverValueType    getResidualNorm() const    {return m_rNorm;}
	virtual SolverValueType    getRelResidualNorm() const {return m_rNorm / m_rhsNorm;}

private:
	int              m_maxIterations;
	float            m_iterations;

    // For stagnation
    int              m_maxmSteps;
    int              m_stag;
    int              m_moreSteps;
    int              m_maxStagSteps;

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
BiCGStabLMonitor<SolverVector>::BiCGStabLMonitor(const int              maxIterations,
                                                 const int              maxmSteps,
                                                 const SolverValueType  relTol,
                                                 const SolverValueType  absTol)
:	m_rNorm(std::numeric_limits<SolverValueType>::max()),
	m_maxIterations(maxIterations),
    m_maxmSteps(maxmSteps),
    m_stag(0),
    m_moreSteps(0),
    m_maxStagSteps(3),
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
BiCGStabLMonitor<SolverVector>::init(const SolverVector& rhs)
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
BiCGStabLMonitor<SolverVector>::finished(const SolverVector& r)
{
	return finished(cusp::blas::nrm2(r));
}

template <typename SolverVector>
inline bool
BiCGStabLMonitor<SolverVector>::finished(SolverValueType rNorm)
{
	m_rNorm = rNorm;

	if (m_code != 0) {
		return true;
    }

	if (isnan(m_rNorm))                      stop(-2, "Residual norm is NaN");
	else if (m_rNorm <= getTolerance())      stop( 1, "Converged");
	else if (m_iterations > m_maxIterations) stop(-1, "Maximum number of iterations was reached");
    else {
        if (m_stag >= m_maxStagSteps && m_moreSteps == 0) {
            m_stag = 0;
        }
        m_moreSteps ++;

        if (m_moreSteps >= m_maxmSteps) {
            stop(-3, "Stagnation encountered");
        }
    }

	return m_code != 0;
}

template <typename SolverVector>
bool
BiCGStabLMonitor<SolverVector>::needCheckConvergence(const SolverVector& r) {
	return needCheckConvergence(cusp::blas::nrm2(r));
}

template <typename SolverVector>
bool
BiCGStabLMonitor<SolverVector>::needCheckConvergence(SolverValueType rNorm) {
	m_rNorm = rNorm;

    if (m_code != 0) {
        return false;
    }
	if (isnan(m_rNorm)) {
        stop(-2, "Residual norm is NaN");
    }

    return m_code == 0 && (m_rNorm <= getTolerance() || m_stag >= m_maxStagSteps || m_moreSteps);
}

} // namespace sap


#endif
