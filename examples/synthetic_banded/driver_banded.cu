#include <algorithm>
#include <fstream>
#include <cmath>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/blas.h>

#include <spike/solver.h>
#include <spike/spmv.h>


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

typedef typename spike::Solver<Vector, PREC_REAL>                 SpikeSolver;
typedef typename spike::SpmvCusp<Matrix>                          SpmvFunctor;

typedef typename cusp::coo_matrix<int, REAL, cusp::host_memory>   MatrixCooH;
typedef typename cusp::array1d<REAL, cusp::host_memory>           VectorH;


// -----------------------------------------------------------------------------
using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::vector;


// -----------------------------------------------------------------------------
// Definitions for SimpleOpt and SimpleGlob
// -----------------------------------------------------------------------------
#include <SimpleOpt/SimpleOpt.h>

// ID values to identify command line arguments
enum {OPT_HELP, OPT_VERBOSE, OPT_PART,
      OPT_TOL, OPT_MAXIT,
      OPT_DROPOFF_FRAC, 
      OPT_BAND,
      OPT_OUTFILE, OPT_PRECOND,
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
	{ OPT_BAND,          "-b",                   SO_MULTI   },
	{ OPT_BAND,          "--banded-synthetic",   SO_MULTI   },
	{ OPT_OUTFILE,       "-o",                   SO_REQ_CMB },
	{ OPT_OUTFILE,       "--output-file",        SO_REQ_CMB },
	{ OPT_PRECOND,       "--precond-method",     SO_REQ_CMB },
	{ OPT_SAFE_FACT,     "--safe-fact",          SO_NONE    },
	{ OPT_VERBOSE,       "-v",                   SO_NONE    },
	{ OPT_VERBOSE,       "--verbose",            SO_NONE    },
	{ OPT_HELP,          "-?",                   SO_NONE    },
	{ OPT_HELP,          "-h",                   SO_NONE    },
	{ OPT_HELP,          "--help",               SO_NONE    },
	SO_END_OF_OPTIONS
};


// -----------------------------------------------------------------------------
// Forward declarations.
// -----------------------------------------------------------------------------
void ShowUsage();
void spikeSetDevice();
bool GetProblemSpecs(int             argc, 
                     char**          argv,
                     int&            N,
                     int&            k,
                     REAL&           d,
                     string&         fileSol,
                     int&            numPart,
                     bool&           verbose,
                     spike::Options& opts);

void GetBandedMatrix(int N, int k, REAL d, Matrix& A);
void GetRhsVector(const Matrix& A, Vector& b, Vector& x_target);
void PrintStats(bool               success,
                const SpikeSolver& mySolver,
                const SpmvFunctor& mySpmv);


// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) 
{
	// Set up the problem to be solved.
	int            pN;
	int            pk;
	REAL           pd;
	string         fileSol;
	int            numPart;
	bool           verbose;
	spike::Options opts;

	opts.trackReordering = false;
	opts.singleComponent = true;
	opts.variableBandwidth = false;
	opts.factMethod = spike::LU_UL;
	opts.performReorder = false;
	opts.applyScaling = false;

	if (!GetProblemSpecs(argc, argv, pN, pk, pd, fileSol, numPart, verbose, opts))
		return 1;

	// Get the device with most available memory.
	spikeSetDevice();

	// Get matrix and rhs.
	Matrix A;
	Vector b;
	Vector x_target;
	Vector delta_x_target;

	GetBandedMatrix(pN, pk, pd, A);
	GetRhsVector(A, b, x_target);

	// Create the SPIKE Solver object and the SPMV functor. Perform the solver
	// setup, then solve the linear system using a 0 initial guess.
	// Set the initial guess to the zero vector.
	SpikeSolver  mySolver(numPart, opts);
	SpmvFunctor  mySpmv(A);
	Vector x(A.num_rows, 0);

	mySolver.setup(A);

	bool success = mySolver.solve(mySpmv, b, x);

	// Write solution file and print solver statistics.
	if (fileSol.length() > 0)
		cusp::io::write_matrix_market_file(x, fileSol);

	// Calculate the actual residual and its norm.
	if (verbose) {
		PrintStats(success, mySolver, mySpmv);

		Vector r(A.num_rows);
		mySpmv(x, r);
		cusp::blas::axpby(b, r, r, REAL(1.0), REAL(-1.0));
		cout << "|b - A*x|      = " << cusp::blas::nrm2(r) << endl;
		cout << "|b|            = " << cusp::blas::nrm2(b) << endl;	
		cout << "|x_target|     = " << cusp::blas::nrm2(x_target) << endl;
		delta_x_target.resize(A.num_rows);
		cusp::blas::axpby(x_target, x, delta_x_target, REAL(1.0), REAL(-1.0));
		cout << "|x_target - x| = " << cusp::blas::nrm2(delta_x_target) << endl;
	} else {
		spike::Stats stats = mySolver.getStats();
		printf("%d,%d,%d,%g,%g\n", success, pN, pk, pd, stats.timeSetup + stats.timeSolve);
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

	// std::cerr << "Use device: " << max_idx << endl;
	// cudaSetDevice(max_idx);
	cudaSetDevice(max_idx);
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

// -----------------------------------------------------------------------------
// GetProblemSpecs()
//
// This function parses the specified program arguments and sets up the problem
// to be solved.
// -----------------------------------------------------------------------------
bool
GetProblemSpecs(int             argc, 
                char**          argv,
                int&            N,
                int&            k,
                REAL&           d,
                string&         fileSol,
                int&            numPart,
                bool&           verbose,
                spike::Options& opts)
{
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
			case OPT_BAND:
				{
					char **mArgs = args.MultiArg(3);
					if (!mArgs) {
						return false;
					}
					N = atoi(mArgs[0]);
					k = atoi(mArgs[1]);
					d = atof(mArgs[2]);

					break;
				}
			case OPT_OUTFILE:
				fileSol = args.OptionArg();
				break;
			case OPT_VERBOSE:
				verbose = true;
				break;
			case OPT_PRECOND:
				{
					string precond = args.OptionArg();
					std::transform(precond.begin(), precond.end(), precond.begin(), ::toupper);
					if (precond == "0" || precond == "SPIKE")
						opts.precondType = spike::Spike;
					else if(precond == "1" || precond == "BLOCK")
						opts.precondType = spike::Block;
					else
						return false;
				}
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

	// If no reordering, force using constant bandwidth.
	if (!opts.performReorder)
		opts.variableBandwidth = false;

	// If using variable bandwidth, force using LU factorization.
	if (opts.variableBandwidth)
		opts.factMethod = spike::LU_only;

	// Print out the problem specifications.
	if (verbose) {
		cout << endl;
		cout << "Problem size: " << N << " "<<k <<" "<<d<<endl;
		if (fileSol.length() > 0)
			cout << "Sol file:    " << fileSol << endl;
		cout << "Using " << numPart << (numPart ==1 ? " partition." : " partitions.") << endl;
		cout << "Iterative solver: " << (opts.solverType == spike::BiCGStab2 ? "BiCGStab2" : "BiCGStab") << endl;
		cout << "Tolerance: " << opts.tolerance << endl;
		cout << "Max. iterations: " << opts.maxNumIterations << endl;
		cout << "Preconditioner: " << (opts.precondType == spike::Spike ? "SPIKE" : "BLOCK DIAGONAL") << endl;
		cout << "Factorization method: LU - UL" << endl;
		if (opts.dropOffFraction > 0)
			cout << "Drop-off fraction: " << opts.dropOffFraction << endl;
		else
			cout << "No drop-off." << endl;
		cout << (opts.safeFactorization ? "Use safe factorization." : "Use non-safe fast factorization.") << endl;
		cout << endl << endl;
	}

	return true;
}


// -----------------------------------------------------------------------------
// ShowUsage()
//
// This function displays the correct usage of this program
// -----------------------------------------------------------------------------
void ShowUsage()
{
	cout << "Usage:  driver_mm [OPTIONS]" << endl;
	cout << endl;
	cout << " -p=NUM_PARTITIONS" << endl;
	cout << " --num-partitions=NUM_PARTITIONS" << endl;
	cout << "        Specify the number of partitions (default 1)." << endl;
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
	cout << " -b SIZE BW DD" << endl;
	cout << " --banded-synthetic SIZE BW DD" << endl;
	cout << "        Use a synthetic banded matrix of size SIZE, half-bandwidth BW," << endl;
	cout << "        and degree of diagonal dominance DD." << endl;
	cout << " -o=OUTFILE" << endl;
	cout << " --output-file=OUTFILE" << endl;
	cout << "        Write the solution to the file OUTFILE (MatrixMarket format)." << endl;
	cout << " --safe-fact" << endl;
	cout << "        Use safe LU-UL factorization." << endl; 
	cout << " --precond-method=METHOD" << endl;
	cout << "        Specify the preconditioner to be used" << endl;
	cout << "        METHOD=0 or METHOD=SPIKE                for using SPIKE preconditioner.  This is the default." << endl;
	cout << "        METHOD=1 or METHOD=BLOCK                for using Block preconditionera." << endl;
	cout << " -? -h --help" << endl;
	cout << "        Print this message and exit." << endl;
	cout << endl;
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


// -----------------------------------------------------------------------------
// PrintStats()
//
// This function prints solver statistics.
// -----------------------------------------------------------------------------
void PrintStats(bool               success,
                const SpikeSolver& mySolver,
                const SpmvFunctor& mySpmv)
{
	spike::Stats stats = mySolver.getStats();

	cout << endl;
	cout << (success ? "Success" : "Failed") << endl;

	cout << "Number of iterations = " << stats.numIterations << endl;
	cout << "Residual norm        = " << stats.residualNorm << endl;
	cout << "Rel. residual norm   = " << stats.relResidualNorm << endl;
	cout << endl;
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
	cout << "SPMV count = " << mySpmv.getCount() 
		 << "  total time = " << mySpmv.getTime() 
		 << "  GFlop/s = " << mySpmv.getGFlops()
		 << endl;
	cout << endl;
}
