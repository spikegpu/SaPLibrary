/** \file banded_matrix.h
 *  \brief Definition of banded matrix class.
 */

#ifndef SAP_BANDED_MATRIX
#define SAP_BANDED_MATRIX

#include <algorithm>
#include <cmath>

#include <cusp/array1d.h>
#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif

namespace sap {

/**
 * This class defines the banded matrix.
 * \tparam Array is the array type used to store the matrix.
 *         (In most cases the same with the solver's array type)
 */
template <typename Array>
class BandedMatrix {
public:
    BandedMatrix(int n, int k, double d);
    typedef typename Array::value_type    value_type;
    typedef typename Array::memory_space  memory_space;
    typedef int                           index_type;

    int m_n;
    int m_k;
    int num_rows;
    int num_cols;
    size_t num_entries;

private:
    typedef typename Array::value_type    ValueType;
    typedef typename Array::memory_space  MemorySpace;

    typedef typename cusp::array1d<ValueType, MemorySpace>        Vector;
    typedef typename cusp::array1d<ValueType, cusp::host_memory>  VectorH;

    Vector     m_A;
    double     m_d;

public:
    const Vector& getBandedMatrix() const {return m_A;}

};

// Banded Matrix constructor.
/**
 * This is the constructor for the BandedMatrix class. It specifies
 * the dimension N, the half-bandwidth K and the diagonal dominance d
 * of the banded matrix. Then it randomly generates a banded matrix
 * with parameters (N, K, d).
 */
template <typename Array>
BandedMatrix<Array>::BandedMatrix(
    int    n,
    int    k,
    double d
) {
    num_rows = num_cols = m_n = n;
    m_k = k;
    m_d = d;
    num_entries = size_t(n) * (2 * k + 1) - k * (k + 1);

    VectorH Ah(n * (2 * k + 1), ValueType(0));

    int middle_index = k;

	for (int ir = 0; ir < n; ir++) {
		int left = std::max(0, ir - k);
		int right = std::min(n - 1, ir + k);

		ValueType row_sum = 0;
		for (int ic = left; ic <= right; ic++) {
			ValueType val = -10.0 + 20.0 * rand() / RAND_MAX;

			if (ir != ic) {
				row_sum += abs(val);
                Ah[middle_index + ic - ir] = val;
            }
		}
		Ah[middle_index] = d * row_sum;

        middle_index += 2 * k + 1;
	}

    m_A = Ah;
}

} // namespace sap

#endif // SAP_BANDED_MATRIX
