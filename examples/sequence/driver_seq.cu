// -----------------------------------------------------------------------------
// 
// -----------------------------------------------------------------------------
#include <algorithm>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/blas.h>

#include <spike/solver.h>


// -----------------------------------------------------------------------------
// Typedefs
// -----------------------------------------------------------------------------
typedef double REAL;

typedef typename cusp::csr_matrix<int, REAL, cusp::device_memory> Matrix;
typedef typename cusp::array1d<REAL, cusp::device_memory>         Vector;

typedef typename spike::Solver<Matrix, Vector> SpikeSolver;


// -----------------------------------------------------------------------------
using std::cout;
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
      OPT_SAFE_FACT,
      OPT_SINGLE_COMP};

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
	{ OPT_SINGLE_COMP,   "--single-component",   SO_NONE    },
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
class CustomSpmv {
public:
	CustomSpmv(Matrix& A) : m_A(A) {}

	void operator()(const Vector& v,
	                Vector&       Av) {cusp::multiply(m_A, v, Av);}

private:
	Matrix&      m_A;
};


// -----------------------------------------------------------------------------
// Forward declarations.
// -----------------------------------------------------------------------------
void ShowUsage();

void spikeSetDevice();

bool GetProblemSpecs(int             argc, 
                     char**          argv,
                     string&         fileMat,
                     int&            numPart,
                     spike::Options& opts);

void PrintStats(bool                success,
                const spike::Stats& stats);



// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) 
{
	// Set up the problem to be solved.
	string         fileMat;
	int            numPart;
	spike::Options opts;

	if (!GetProblemSpecs(argc, argv, fileMat, numPart, opts))
		return 1;
	
	// Get the device with most available memory.
	spikeSetDevice();

	// Get matrix and rhs.
	Matrix A;
	cusp::io::read_matrix_market_file(A, fileMat);

	// Create the SPIKE Solver object and the custom SPMV functor.
	SpikeSolver mySolver(numPart, opts);
	CustomSpmv  mySpmv(A);

	// Perform the solver setup.
	mySolver.setup(A);

	// Solve the linear system A*x = b for two different RHS.
	// In each case, set the initial guess to 0.
	{
		Vector b(A.num_rows, 1.0);
		Vector x(A.num_rows, 0.0);
		mySolver.solve(mySpmv, b, x);
		////cusp::io::write_matrix_market_file(x, "x1.mtx");
	}

	{
		Vector b(A.num_rows, 2.0);
		Vector x(A.num_rows, 0.0);
		mySolver.solve(mySpmv, b, x);
		////cusp::io::write_matrix_market_file(x, "x2.mtx");
	}

	// Perturb the non-zero entries in the A matrix and update the Spike solver.
	// Then solve again the linear system A*x = b twice.
	cusp::blas::scal(A.values, 1.1);
	mySolver.update(A.values);

	{
		Vector b(A.num_rows, 1.0);
		Vector x(A.num_rows, 0.0);
		mySolver.solve(mySpmv, b, x);
		////cusp::io::write_matrix_market_file(x, "y1.mtx");
	}

	{
		Vector b(A.num_rows, 2.0);
		Vector x(A.num_rows, 0.0);
		mySolver.solve(mySpmv, b, x);
		////cusp::io::write_matrix_market_file(x, "y2.mtx");
	}


	return 0;
}

// -----------------------------------------------------------------------------
// spikeSetDevice()
//
// This function sets the active device to be the one with maximum available
// space.
// -----------------------------------------------------------------------------
void spikeSetDevice() {
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
                spike::Options& opts)
{
	opts.solverType = spike::BiCGStab2;
	opts.precondMethod = spike::Spike;
	opts.method = spike::LU_only;
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
				opts.tolerance = atof(args.OptionArg());
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
			case OPT_SINGLE_COMP:
				opts.singleComponent = true;
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
	cout << "Tolerance: " << opts.tolerance << endl;
	cout << "Max. iterations: " << opts.maxNumIterations << endl;
	if (opts.dropOffFraction > 0)
		cout << "Drop-off fraction: " << opts.dropOffFraction << endl;
	else
		cout << "No drop-off." << endl;
	cout << (opts.singleComponent ? "Do not break the problem into several components." : "Attempt to break the problem into several components.") << endl;
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
	cout << "Usage:  driver_seq [OPTIONS]" << endl;
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
	cout << "        Frobenius norm is ignored (default 0.0 -- i.e. no drop-off)." << endl;
	cout << " --single-component" << endl;
	cout << "        Do not break the problem into several components." << endl;
	cout << " --safe-fact" << endl;
	cout << "        Use safe LU-UL factorization." << endl; 
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
                const spike::Stats& stats)
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
