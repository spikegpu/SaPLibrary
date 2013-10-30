#ifndef SPIKE_EXCEPTION_H
#define SPIKE_EXCEPTION_H

#include <exception>

namespace spike {

class NegativeReducedWeightException: public std::exception
{
	virtual const char* what() const throw() {return "Negative reduced weight found in MC64";}
};

}

#endif
