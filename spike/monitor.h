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
template <typename Vector>
class Monitor
{
public:
	typedef typename Vector::value_type   ValueType;

	enum State {
		Continue,
		Converged,
		Failed
	};

	Monitor(const int          maxIterations,
	        const ValueType    tolerance);
	~Monitor() {}

	void init(const Vector& rhs);

	bool done(const Vector& r);

	void increment(float incr) {m_iterations += incr;}

	bool         converged() const          {return m_state == Converged;}
	int          getMaxIterations() const   {return m_maxIterations;}
	float        getNumIterations() const   {return m_iterations;}
	ValueType    getTolerance() const       {return m_tolerance;}
	ValueType    getRHSNorm() const         {return m_rhsNorm;}
	ValueType    getResidualNorm() const    {return m_rNorm;}
	ValueType    getRelResidualNorm() const {return m_rNorm / m_rhsNorm;}

private:
	State          m_state;
	int            m_maxIterations;
	float          m_iterations;
	
	ValueType      m_tolerance;
	ValueType      m_rhsNorm;
	ValueType      m_rNorm;
};


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
template <typename Vector>
inline
Monitor<Vector>::Monitor(const int          maxIterations,
                         const ValueType    tolerance)
:	m_rNorm(std::numeric_limits<ValueType>::max()),
	m_maxIterations(maxIterations),
	m_tolerance(tolerance),
	m_iterations(0),
	m_state(Continue)
{
}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
template <typename Vector>
inline void
Monitor<Vector>::init(const Vector& rhs)
{
	m_rhsNorm = cusp::blas::nrm2(rhs);
	m_iterations = 0;
}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
template <typename Vector>
inline bool
Monitor<Vector>::done(const Vector& r)
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
