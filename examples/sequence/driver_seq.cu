// -----------------------------------------------------------------------------
// 
// -----------------------------------------------------------------------------
#include <algorithm>
#include <sap/solver.h>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif



// -----------------------------------------------------------------------------
// Typedefs
// -----------------------------------------------------------------------------
typedef double REAL;
typedef float  PREC_REAL;

typedef typename cusp::csr_matrix<int, REAL, cusp::device_memory> Matrix;
typedef typename cusp::array1d<REAL, cusp::device_memory>         Vector;

typedef typename sap::Solver<Vector, PREC_REAL>                 SaPSolver;


// -----------------------------------------------------------------------------
using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::string;


// -----------------------------------------------------------------------------
// Definitions for argument parsing with SimpleOpt
// -----------------------------------------------------------------------------
#include <SimpleOpt/SimpleOpt.h>


// ID values to identify command line arguments
enum {OPT_HELP, OPT_VERBOSE, OPT_PART,
      OPT_TOL, OPT_MAXIT,
      OPT_DROPOFF_FRAC, 
      OPT_MATFILE,
      OPT_SAFE_FACT};

// Table of CSimpleOpt::Soption structures. Each entry specifies:
// - the ID for the option (returned from OptionId() during processing)
// - the option as it should appear on the command line
// - type of the option
// The last entry must be SO_END_OF_OPTIONS
CSimpleOptA::SOption g_options[] = {
	{ OPT_PART,          "-p",                   SO_REQ_CMB },
	{ OPT_PART,          "--num-partitions",     SO_REQ_CMB },
	{ OPT_TOL,           "-t",                   SO_REQ_CMB },
	{ OPT_TOL,           "--tolerance",          SO_REQ_CMB },
	{ OPT_MAXIT,         "-i",                   SO_REQ_CMB },
	{ OPT_MAXIT,         "--max-num-iterations", SO_REQ_CMB },
	{ OPT_DROPOFF_FRAC,  "-d",                   SO_REQ_CMB },
	{ OPT_DROPOFF_FRAC,  "--drop-off-fraction",  SO_REQ_CMB },
	{ OPT_MATFILE,       "-m",                   SO_REQ_CMB },
	{ OPT_MATFILE,       "--matrix-file",        SO_REQ_CMB },
	{ OPT_SAFE_FACT,     "--safe-fact",          SO_NONE    },
	{ OPT_VERBOSE,       "-v",                   SO_NONE    },
	{ OPT_VERBOSE,       "--verbose",            SO_NONE    },
	{ OPT_HELP,          "-?",                   SO_NONE    },
	{ OPT_HELP,          "-h",                   SO_NONE    },
	{ OPT_HELP,          "--help",               SO_NONE    },
	SO_END_OF_OPTIONS
};


// -----------------------------------------------------------------------------
// CustomSpmv
//
// This class defines a custom SPMV functor for sparse matrix-vector product.
// -----------------------------------------------------------------------------
class CustomSpmv : public cusp::linear_operator<Matrix::value_type, Matrix::memory_space, Matrix::index_type> {
public:
	typedef cusp::linear_operator<Matrix::value_type, Matrix::memory_space, Matrix::index_type> Parent;

	CustomSpmv(Matrix& A) : Parent(A.num_rows, A.num_cols), m_A(A) {}

	void operator()(const Vector& v,
	                Vector&       Av) 
	{
		cusp::multiply(m_A, v, Av);
	}

	Matrix&      m_A;
private:
};


// -----------------------------------------------------------------------------
// Forward declarations.
// -----------------------------------------------------------------------------
void ShowUsage();

void sapSetDevice();

bool GetProblemSpecs(int             argc, 
                     char**          argv,
                     string&         fileMat,
                     int&            numPart,
                     sap::Options& opts);

void PrintStats(bool                success,
                const sap::Stats& stats);



// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) 
{
	// Set up the problem to be solved.
	string         fileMat;
	int            numPart;
	sap::Options opts;

	if (!GetProblemSpecs(argc, argv, fileMat, numPart, opts))
		return 1;
	
	// Get the device with most available memory.
	sapSetDevice();

	// Get matrix and rhs.
	Matrix A;
	cusp::io::read_matrix_market_file(A, fileMat);

	// Create the SAP Solver object and the custom SPMV functor.
	SaPSolver mySolver(numPart, opts);
	CustomSpmv  mySpmv(A);

	// Perform the solver setup.
	mySolver.setup(A);

	bool converged;

	// Solve the linear system A*x = b for two different RHS.
	// In each case, set the initial guess to 0.
	{
		Vector b(A.num_rows, 1.0);
		Vector x(A.num_rows, 0.0);
		converged = mySolver.solve(mySpmv, b, x);
		cout << (converged ? "Converged" : "Not Converged") << endl;
		////cusp::io::write_matrix_market_file(x, "x1.mtx");
	}

	{
		Vector b(A.num_rows, 2.0);
		Vector x(A.num_rows, 0.0);
		converged = mySolver.solve(mySpmv, b, x);
		cout << (converged ? "Converged" : "Not Converged") << endl;
		////cusp::io::write_matrix_market_file(x, "x2.mtx");
	}

	// Perturb the non-zero entries in the A matrix and update the SaP solver.
	// Then solve again the linear system A*x = b twice.
	cusp::blas::scal(A.values, 1.1);
	bool update_success = mySolver.update(A.values);

	if (update_success)
	{
		{
			Vector b(A.num_rows, 1.0);
			Vector x(A.num_rows, 0.0);
			converged = mySolver.solve(mySpmv, b, x);
			cout << (converged ? "Converged" : "Not Converged") << endl;
			////cusp::io::write_matrix_market_file(x, "y1.mtx");
		}

		{
			Vector b(A.num_rows, 2.0);
			Vector x(A.num_rows, 0.0);
			converged = mySolver.solve(mySpmv, b, x);
			cout << (converged ? "Converged" : "Not Converged") << endl;
			////cusp::io::write_matrix_market_file(x, "y2.mtx");
		}
	} else {
		cerr << "Update failed (probably due to matrix pattern change)." << endl;
		return 1;
	}


	return 0;
}

// -----------------------------------------------------------------------------
// sapSetDevice()
//
// This function sets the active device to be the one with maximum available
// space.
// -----------------------------------------------------------------------------
void sapSetDevice() {
	int deviceCount = 0;
	
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
		std::cerr << "There is no available device." << endl;
		exit(-1);
	}

	size_t max_free_size = 0;
	int max_idx = 0;
	for (int i=0; i < deviceCount; i++) {
		cudaSetDevice(i);
		size_t free_size = 0, total_size = 0;
		if (cudaMemGetInfo(&free_size, &total_size) == cudaSuccess)
			if (max_free_size < free_size) {
				max_idx = i;
				max_free_size = free_size;
			}
	}

	std::cerr << "Use device: " << max_idx << endl;
	cudaSetDevice(max_idx);
}


// -----------------------------------------------------------------------------
// GetProblemSpecs()
//
// This function parses the specified program arguments and sets up the problem
// to be solved.
// -----------------------------------------------------------------------------
bool
GetProblemSpecs(int             argc, 
                char**          argv,
                string&         fileMat,
                int&            numPart,
                sap::Options& opts)
{
	opts.solverType = sap::BiCGStab2;
	opts.precondType = sap::Spike;
	opts.factMethod = sap::LU_only;
	opts.performReorder = true;
	opts.applyScaling = true;
	opts.dropOffFraction = 0.0;
	opts.variableBandwidth = true;
	opts.trackReordering = true;

	numPart = -1;

	// Create the option parser and pass it the program arguments and the array
	// of valid options. Then loop for as long as there are arguments to be
	// processed.
	CSimpleOptA args(argc, argv, g_options);

	while (args.Next()) {
		// Exit immediately if we encounter an invalid argument.
		if (args.LastError() != SO_SUCCESS) {
			cout << "Invalid argument: " << args.OptionText() << endl;
			ShowUsage();
			return false;
		}

		// Process the current argument.
		switch (args.OptionId()) {
			case OPT_HELP:
				ShowUsage();
				return false;
			case OPT_PART:
				numPart = atoi(args.OptionArg());
				break;
			case OPT_TOL:
				opts.relTol = atof(args.OptionArg());
				break;
			case OPT_MAXIT:
				opts.maxNumIterations = atoi(args.OptionArg());
				break;
			case OPT_DROPOFF_FRAC:
				opts.dropOffFraction = atof(args.OptionArg());
				break;
			case OPT_MATFILE:
				fileMat = args.OptionArg();
				break;
			case OPT_SAFE_FACT:
				opts.safeFactorization = true;
				break;
		}
	}

	// If the number of partitions was not defined, show usage and exit.
	if (numPart <= 0) {
		cout << "The number of partitions must be specified." << endl << endl;
		ShowUsage();
		return false;
	}

	// If no problem was defined, show usage and exit.
	if (fileMat.length() == 0) {
		cout << "The matrix filename is required." << endl << endl;
		ShowUsage();
		return false;
	}

	// Print out the problem specifications.
	cout << endl;
	cout << "Matrix file: " << fileMat << endl;
	cout << "Using " << numPart << (numPart ==1 ? " partition." : " partitions.") << endl;
	cout << "Relative tolerance: " << opts.relTol << endl;
	cout << "Max. iterations: " << opts.maxNumIterations << endl;
	if (opts.dropOffFraction > 0)
		cout << "Drop-off fraction: " << opts.dropOffFraction << endl;
	else
		cout << "No drop-off." << endl;
	cout << (opts.safeFactorization ? "Use safe factorization." : "Use non-safe fast factorization.") << endl;
	cout << endl << endl;

	return true;
}


// -----------------------------------------------------------------------------
// ShowUsage()
//
// This function displays the correct usage of this program
// -----------------------------------------------------------------------------
void ShowUsage()
{
	cout << "Usage:  driver_seq -p=NUM_PARTITIONS -m=MATFILE [OPTIONS]" << endl;
	cout << endl;
	cout << " -m=MATFILE" << endl;
	cout << " --matrix-file=MATFILE" << endl;
	cout << "        Read the matrix from the MatrixMarket file MATFILE." << endl;
	cout << " -p=NUM_PARTITIONS" << endl;
	cout << " --num-partitions=NUM_PARTITIONS" << endl;
	cout << "        Specify the number of partitions." << endl;
	cout << " -t=TOLERANCE" << endl;
	cout << " --tolerance=TOLERANCE" << endl;
	cout << "        Use TOLERANCE for BiCGStab stopping criteria (default 1e-6)." << endl;
	cout << " -i=ITERATIONS" << endl;
	cout << " --max-num-iterations=ITERATIONS" << endl;
	cout << "        Use at most ITERATIONS for BiCGStab (default 100)." << endl;
	cout << " -d=FRACTION" << endl;
	cout << " --drop-off-fraction=FRACTION" << endl;
	cout << "        Drop off-diagonal elements such that FRACTION of the matrix" << endl;
	cout << "        element-wise 1-norm is ignored (default 0.0 -- i.e. no drop-off)." << endl;
	cout << " --safe-fact" << endl;
	cout << "        Use safe LU-UL factorization (default false)." << endl; 
	cout << " -? -h --help" << endl;
	cout << "        Print this message and exit." << endl;
	cout << endl;
}


// -----------------------------------------------------------------------------
// PrintStats()
//
// This function prints solver statistics.
// -----------------------------------------------------------------------------
void PrintStats(bool                success,
                const sap::Stats& stats)
{
	cout << endl;
	cout << (success ? "Success" : "Failed") << endl;

	cout << "Number of iterations = " << stats.numIterations << endl;
	cout << "Residual norm        = " << stats.residualNorm << endl;
	cout << "Rel. residual norm   = " << stats.relResidualNorm << endl;
	cout << endl;
	cout << "Bandwidth after reordering = " << stats.bandwidthReorder << endl;
	cout << "Bandwidth                  = " << stats.bandwidth << endl;
	cout << "Actual drop-off fraction   = " << stats.actualDropOff << endl;
	cout << endl;
	cout << "Setup time total  = " << stats.timeSetup << endl;
	double timeSetupGPU = stats.time_toBanded + stats.time_offDiags
		+ stats.time_bandLU + stats.time_bandUL
		+ stats.time_assembly + stats.time_fullLU;
	cout << "  Setup time GPU  = " << timeSetupGPU << endl;
	cout << "    form banded matrix       = " << stats.time_toBanded << endl;
	cout << "    extract off-diags blocks = " << stats.time_offDiags << endl;
	cout << "    banded LU factorization  = " << stats.time_bandLU << endl;
	cout << "    banded UL factorization  = " << stats.time_bandUL << endl;
	cout << "    assemble reduced matrix  = " << stats.time_assembly << endl;
	cout << "    reduced matrix LU        = " << stats.time_fullLU << endl;
	cout << "  Setup time CPU  = " << stats.timeSetup - timeSetupGPU << endl;
	cout << "    reorder                  = " << stats.time_reorder << endl;
	cout << "    CPU assemble             = " << stats.time_cpu_assemble << endl;
	cout << "    data transfer            = " << stats.time_transfer << endl;
	cout << "Solve time        = " << stats.timeSolve << endl;
	cout << "  shuffle time    = " << stats.time_shuffle << endl;
	cout << endl;
	cout << endl;
}
