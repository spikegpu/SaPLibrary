// ============================================================================
// This file implements the strided_range class to allow non-contiguous
// access to data in a thrust vector.
//
// This class is the same as the strided_range thrust example provided
// by Nathan Bell at
// (https://code.google.com/p/thrust/source/browse/examples/strided_range.cu)
// ============================================================================


#ifndef SAP_STRIDED_RANGE
#define SAP_STRIDED_RANGE

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>


namespace sap {

template <typename Iterator>
class strided_range
{
public:
	typedef typename thrust::iterator_difference<Iterator>::type difference_type;

	struct stride_functor : public thrust::unary_function<difference_type, difference_type>
	{
		stride_functor(difference_type stride) : m_stride(stride) {}

		__host__ __device__
		difference_type operator()(const difference_type& i) const {return m_stride * i;}

		difference_type m_stride;
	};

	typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
	typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
	typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

	// Type of the strided_range iterator
	typedef PermutationIterator iterator;

	// Construct strided_range for the range [first,last)
	strided_range(Iterator first, Iterator last, difference_type stride)
		: m_first(first), m_last(last), m_stride(stride) {}
	
	iterator begin(void) const
	{
		return PermutationIterator(m_first, TransformIterator(CountingIterator(0), stride_functor(m_stride)));
	}

	iterator end(void) const
	{
		return begin() + ((m_last - m_first) + (m_stride - 1)) / m_stride;
	}

protected:
	Iterator        m_first;
	Iterator        m_last;
	difference_type m_stride;
};

} // namespace sap



#endif
