// -----------------------------------------------------------------------------
// This example shows how to use views to a CUSP array in conjunctin with the
// SPIKE solver.
//
// As an illustration, we solve a system of the form
//
// [A  0] [x1] = [b1]
// [0  A] [x2]   [b2]
//
// where b1 and b2 are views in the first half and second half of some array b,
// respectively. Similarly, x1 and x2 are views in a larger array x.
//
// Usage:
//    driver_views FILENAME NUMPART
// where FILENAME is the name of a matrix market file defining the matrix A and
// NUMPART is the number of partitions to be used.
// -----------------------------------------------------------------------------
#include <cstdlib>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif

#include <spike/solver.h>
#include <spike/spmv.h>


// -----------------------------------------------------------------------------
// Typedefs
// -----------------------------------------------------------------------------
typedef double REAL;
typedef float  PREC_REAL;

typedef typename cusp::csr_matrix<int, REAL, cusp::device_memory> Matrix;
typedef typename cusp::array1d<REAL, cusp::device_memory>         Vector;
typedef typename Vector::view                                     VectorView;

typedef typename spike::Solver<VectorView, PREC_REAL>             SpikeSolver;
typedef typename spike::SpmvCusp<Matrix>                          SpmvFunctor;


// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) 
{
	// Get the name of a matrix market file from the command argument
	if (argc < 3) {
		std::cerr << "Usage" << argv[0] << " FILENAME NUMPART" << std::endl;
		return 1;
	}

	std::string filename(argv[1]);
	int         num_part = atoi(argv[2]);

	// Get matrix
	Matrix A;
	cusp::io::read_matrix_market_file(A, filename);

	size_t n = A.num_rows;

	// Create the extended RHS vector and two views into it.
	// Fill the first half with 1s and the second half with 2s.
	Vector        b(2 * n);
	VectorView    b1(b.begin(), b.begin() + n);
	VectorView    b2(b.begin() + n, b.end());

	cusp::blas::fill(b1, 1);
	cusp::blas::fill(b2, 2);

	// Create the extended solution vector, initialized to 0, and
	// two views into it.
	Vector        x(2 * n, 0);
	VectorView  x1(x.begin(), x.begin() + n);
	VectorView  x2(x.begin() + n, x.end());

	// Create the SPIKE Solver object and the SPMV functor and perform the
	// solver setup.
	spike::Options  opts;
	SpikeSolver     mySolver(num_part, opts);
	SpmvFunctor     mySpmv(A);

	mySolver.setup(A);

	// Solve the linear systems A*x1 = b1 and A*x2 = b2.
	mySolver.solve(mySpmv, b1, x1);
	mySolver.solve(mySpmv, b2, x2);

	////cusp::io::write_matrix_market_file(x1, "x1.mtx");
	////cusp::io::write_matrix_market_file(x2, "x2.mtx");
	////cusp::io::write_matrix_market_file(x,  "x.mtx");

	return 0;
}
