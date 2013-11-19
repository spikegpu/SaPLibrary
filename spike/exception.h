#ifndef SPIKE_EXCEPTION_H
#define SPIKE_EXCEPTION_H

#include <stdexcept>
#include <string>

namespace spike {

class system_error : public std::runtime_error
{
public:
	enum Reason
	{
		Zero_pivoting        = -1,
		Negative_MC64_weight = -2,
		Illegal_update       = -3,
		Illegal_solve        = -4,
		Matrix_singular      = -5
	};

	system_error(Reason             reason,
	             const std::string& what_arg)
	: std::runtime_error(what_arg),
	  m_reason(reason)
	{}

	system_error(Reason      reason,
	             const char* what_arg)
	: std::runtime_error(what_arg),
	  m_reason(reason)
	{}
	
	virtual ~system_error() throw() {}

	Reason  reason() const {return m_reason;}

private:
	Reason        m_reason;
};

} // namespace spike


#endif
