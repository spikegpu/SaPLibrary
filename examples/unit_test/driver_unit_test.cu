#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cusp/array1d.h>
#include <sap/common.h>
#include <sap/solver.h>
#include <sap/spmv.h>

#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>

using ::testing::Return;

// -----------------------------------------------------------------------------
// Macro to obtain a random number between two specified values
// -----------------------------------------------------------------------------
#define RAND(L,H)  ((L) + ((H)-(L)) * (float)rand()/(float)RAND_MAX)

// -----------------------------------------------------------------------------
// Typedefs
// -----------------------------------------------------------------------------
typedef double REAL;
typedef float  PREC_REAL;

typedef typename cusp::csr_matrix<int, REAL, cusp::device_memory> Matrix;
typedef typename cusp::array1d<REAL, cusp::device_memory>         Vector;

typedef typename sap::SpmvCusp<Matrix>                          SpmvFunctor;

typedef typename cusp::coo_matrix<int, REAL, cusp::host_memory>   MatrixCooH;
typedef typename cusp::array1d<REAL, cusp::host_memory>           VectorH;

void GetBandedMatrix(int N, int k, REAL d, Matrix& A);
void GetRhsVector(const Matrix& A, Vector& b, Vector& x_target);

class MockSaPSolver: public sap::Solver<Vector, PREC_REAL> {
public:
    MockSaPSolver(int numPart, const sap::Options& opts): sap::Solver<Vector, PREC_REAL>(numPart, opts) {
    }
    // MOCK_METHOD1(setup, bool(const Matrix&));
    // MOCK_CONST_METHOD0(getMonitorMessage, std::string&());
};

TEST(DenseBandedTest, OverallTest) {
    Matrix A;
    Vector x_target;
    Vector b;

    int pN = 10000;
    int pk = 20;
    int numPart = 10;
    REAL pd = 1.0;

	GetBandedMatrix(pN, pk, pd, A);
	GetRhsVector(A, b, x_target);

	sap::Options opts;

	opts.trackReordering = false;
	opts.variableBandwidth = false;
	opts.factMethod = sap::LU_UL;
	opts.performReorder = false;
	opts.applyScaling = false;
    opts.relTol = 1e-13;

	// Create the SAP Solver object and the SPMV functor. Perform the solver
	// setup, then solve the linear system using a 0 initial guess.
	// Set the initial guess to the zero vector.

	MockSaPSolver  mySolver(numPart, opts);
	SpmvFunctor  mySpmv(A);
	Vector x(A.num_rows, 0);

	mySolver.setup(A);
    bool success = mySolver.solve(mySpmv, b, x);

    VectorH xh = x;
    VectorH xh_target = x_target;

    REAL max_val = cusp::blas::nrmmax(x_target);

    ASSERT_EQ(xh.size(), xh_target.size());

    EXPECT_TRUE(success);
    EXPECT_EQ(1, mySolver.getMonitorCode());
    EXPECT_EQ("Converged", mySolver.getMonitorMessage());
    EXPECT_DOUBLE_EQ(1e-13, opts.relTol);
    EXPECT_GE(1e-13, mySolver.getStats().relResidualNorm);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// -------------------------------------------------------------------
// GetBandedMatrix()
//
// This function generates a banded matrix of specified size, half
// bandwidth, and degree of diagonal dominance. The matrix is first
// generated on a local COO matrix on the host and is then copied to
// the output matrix. We use random elements in the range [-10, 10]
// and adjust the diagonal elements to satisfy the required degree of
// diagonal dominance.
// -------------------------------------------------------------------
void
GetBandedMatrix(int N, int k, REAL d, Matrix& A)
{
	// Generate the banded matrix (in COO format) on the host.
	int     num_entries = (2 * k + 1) * N - k * (k + 1);
	MatrixCooH Ah(N, N, num_entries);

	int iiz = 0;
	for (int ir = 0; ir < N; ir++) {
		int left = std::max(0, ir - k);
		int right = std::min(N - 1, ir + k);

		REAL row_sum = 0;
		int  diag_iiz;
		for (int ic = left; ic <= right; ic++, iiz++) {
			REAL val = RAND(-10.0, 10.0);////(ir+1)*(ic+1);

			if (ir == ic)
				diag_iiz = iiz;
			else
				row_sum += abs(val);

			Ah.row_indices[iiz] = ir;
			Ah.column_indices[iiz] = ic;
			Ah.values[iiz] = val;
		}
		Ah.values[diag_iiz] = d * row_sum;
	}

	// Copy the matrix from host to device, while also converting it 
	// from COO to CSR format.
	A = Ah;

	////cusp::io::write_matrix_market_file(Ah, "A.mtx");
}

// -------------------------------------------------------------------
// GetRhsVector()
//
// This function generates a RHS vector of appropriate dimension. We
// use the method of manufactured solution, meaning we set
//    b = A * x
// for a known "solution" vector x.
// -------------------------------------------------------------------
void
GetRhsVector(const Matrix& A, Vector& b, Vector& x_target)
{
	// Create a desired solution vector (on the host), then copy it
	// to the device.
	int     N = A.num_rows;
	REAL    dt = 1.0/(N-1);
	REAL    max_val = 100.0;

	VectorH xh(N);

	for (int i = 0; i < N; i++) {
		REAL t = i *dt;
		xh[i] = 4 * max_val * t * (1 - t);
	}

	x_target = xh;
	
	// Calculate the RHS vector.
	b.resize(N);
	cusp::multiply(A, x_target, b);
	////cusp::io::write_matrix_market_file(b, "b.mtx");
}
