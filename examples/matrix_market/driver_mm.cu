#include <algorithm>
#include <fstream>

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
typedef typename cusp::array1d<REAL, cusp::host_memory>           VectorHost;

typedef typename spike::Solver<Matrix, Vector>       SpikeSolver;
typedef typename spike::SpmvCusp<Matrix, Vector>     SpmvFunctor;


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
	  OPT_NO_REORDERING, OPT_NO_SCALING,
	  OPT_TOL, OPT_MAXIT,
	  OPT_DROPOFF_FRAC, 
	  OPT_MATFILE, OPT_RHSFILE, 
	  OPT_OUTFILE, OPT_FACTORIZATION, OPT_PRECOND,
	  OPT_KRYLOV, OPT_SAFE_FACT,
	  OPT_CONST_BAND, OPT_SINGLE_COMP};

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
	{ OPT_RHSFILE,       "-r",                   SO_REQ_CMB },
	{ OPT_RHSFILE,       "--rhs-file",           SO_REQ_CMB },
	{ OPT_OUTFILE,       "-o",                   SO_REQ_CMB },
	{ OPT_OUTFILE,       "--output-file",        SO_REQ_CMB },
	{ OPT_SINGLE_COMP,	 "--single-component",	 SO_NONE	},
	{ OPT_NO_REORDERING, "--no-reordering",      SO_NONE    },
	{ OPT_NO_SCALING,    "--no-scaling",         SO_NONE    },
	{ OPT_FACTORIZATION, "-f",                   SO_REQ_CMB },
	{ OPT_FACTORIZATION, "--factorization-method", SO_REQ_CMB },
	{ OPT_PRECOND,		 "--precond-method",	 SO_REQ_CMB },
	{ OPT_KRYLOV,        "-k",                   SO_REQ_CMB },
	{ OPT_KRYLOV,        "--krylov-method",      SO_REQ_CMB },
	{ OPT_SAFE_FACT,     "--safe-fact",          SO_NONE    },
	{ OPT_CONST_BAND,    "--const-band",         SO_NONE    },
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
bool GetProblemSpecs(int                   argc, 
                     char**                argv,
				     string&               fileMat,
                     string&               fileRhs,
                     string&               fileSol,
                     int&                  numPart,
                     spike::SolverOptions& opts);
void PrintStats(bool               success,
                const SpikeSolver& mySolver,
                const SpmvFunctor& mySpmv);


// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) 
{
	// Set up the problem to be solved.
	string  fileMat;
	string  fileRhs;
	string  fileSol;
	int     numPart;
	spike::SolverOptions  opts;

	if (!GetProblemSpecs(argc, argv, fileMat, fileRhs, fileSol, numPart, opts))
		return 1;

	// Get the device with most available memory.
	spikeSetDevice();

	// Get matrix and rhs.
	Matrix A;
	Vector b;

	cusp::io::read_matrix_market_file(A, fileMat);

	if (fileRhs.length() > 0)
		cusp::io::read_matrix_market_file(b, fileRhs);
	else
		b.resize(A.num_rows, 1);

	// Create the SPIKE Solver object and the SPMV functor. Perform the solver
	// setup, then solve the linear system using a 0 initial guess.
	// Set the initial guess to the zero vector.
	SpikeSolver mySolver(numPart, opts);
	SpmvFunctor  mySpmv(A);
	Vector       x(A.num_rows, 0);

	mySolver.setup(A);
	bool success = mySolver.solve(mySpmv, b, x);

	// Write solution file and print solver statistics.
	if (fileSol.length() > 0)
		cusp::io::write_matrix_market_file(x, fileSol);

	PrintStats(success, mySolver, mySpmv);

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
GetProblemSpecs(int                   argc, 
                char**                argv,
				string&               fileMat,
                string&               fileRhs,
                string&               fileSol,
                int&                  numPart,
                spike::SolverOptions& opts)
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
			case OPT_NO_REORDERING:
				opts.performReorder = false;
				break;
			case OPT_NO_SCALING:
				opts.applyScaling = false;
				break;
			case OPT_MATFILE:
				fileMat = args.OptionArg();
				break;
			case OPT_RHSFILE:
				fileRhs = args.OptionArg();
				break;
			case OPT_OUTFILE:
				fileSol = args.OptionArg();
				break;
			case OPT_FACTORIZATION:
				{
					string fact = args.OptionArg();
					std::transform(fact.begin(), fact.end(), fact.begin(), ::toupper);
					if (fact == "0" || fact == "LU_UL")
						opts.method = spike::LU_UL;
					else if (fact == "1" || fact == "LU_LU")
						opts.method = spike::LU_only;
					else
						return false;
				}
				break;
			case OPT_PRECOND:
				{
					string precond = args.OptionArg();
					std::transform(precond.begin(), precond.end(), precond.begin(), ::toupper);
					if (precond == "0" || precond == "SPIKE")
						opts.precondMethod = spike::Spike;
					else if(precond == "1" || precond == "BLOCK")
						opts.precondMethod = spike::Block;
					else
						return false;
				}
				break;
			case OPT_KRYLOV:
				{
					string kry = args.OptionArg();
					std::transform(kry.begin(), kry.end(), kry.begin(), ::toupper);
					if (kry == "0" || kry == "BICGSTAB")
						opts.solverType = spike::BiCGStab;
					else if (kry == "1" || kry == "BICGSTAB2")
						opts.solverType = spike::BiCGStab2;
					else
						return false;
				}
				break;
			case OPT_SINGLE_COMP:
				opts.singleComponent = true;
				break;
			case OPT_SAFE_FACT:
				opts.safeFactorization = true;
				break;
			case OPT_CONST_BAND:
				opts.variousBandwidth = false;
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

	// If no reordering, force using constant bandwidth.
	if (!opts.performReorder)
		opts.variousBandwidth = false;

	// If using variable bandwidth, force using LU factorization.
	if (opts.variousBandwidth)
		opts.method = spike::LU_only;

	// Print out the problem specifications.
	cout << endl;
	cout << "Matrix file: " << fileMat << endl;
	if (fileRhs.length() > 0)
		cout << "Rhs file:    " << fileRhs << endl;
	if (fileSol.length() > 0)
		cout << "Sol file:    " << fileSol << endl;
	cout << "Using " << numPart << (numPart ==1 ? " partition." : " partitions.") << endl;
	cout << "Iterative solver: " << (opts.solverType == spike::BiCGStab2 ? "BiCGStab2" : "BiCGStab") << endl;
	cout << "Tolerance: " << opts.tolerance << endl;
	cout << "Max. iterations: " << opts.maxNumIterations << endl;
	cout << "Preconditioner: " << (opts.precondMethod == spike::Spike ? "SPIKE" : "BLOCK DIAGONAL") << endl;
	cout << "Factorization method: " << (opts.method == spike::LU_UL ? "LU - UL" : "LU - LU") << endl;
	if (opts.dropOffFraction > 0)
		cout << "Drop-off fraction: " << opts.dropOffFraction << endl;
	else
		cout << "No drop-off." << endl;
	cout << (opts.singleComponent ? "Do not break the problem into several components." : "Attempt to break the problem into several components.") << endl;
	cout << (opts.performReorder ? "Perform reordering." : "Do not perform reordering.") << endl;
	cout << (opts.applyScaling ? "Apply scaling." : "Do not apply scaling.") << endl;
	cout << (opts.safeFactorization ? "Use safe factorization." : "Use non-safe fast factorization.") << endl;
	cout << (opts.variousBandwidth ? "Use variable bandwidth method." : "Use constant bandwidth method.") << endl;
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
	cout << "Usage:  driver_mm [OPTIONS]" << endl;
	cout << endl;
	cout << " -p=NUM_PARTITIONS" << endl;
	cout << " --num-partitions=NUM_PARTITIONS" << endl;
	cout << "        Specify the number of partitions (default 1)." << endl;
	cout << " --no-reordering" << endl;
	cout << "        Do not perform reordering." << endl;
	cout << " --no-scaling" << endl;
	cout << "        Do not perform scaling (ignored if --no-reordering is specified)" << endl;
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
	cout << " -m=MATFILE" << endl;
	cout << " --matrix-file=MATFILE" << endl;
	cout << "        Read the matrix from the file MATFILE (MatrixMarket format)." << endl;
	cout << " -r=RHSFILE" << endl;
	cout << " --rhs-file=RHSFILE" << endl;
	cout << "        Read the right-handside vector from the file RHSFILE (MatrixMarket format)." << endl;
	cout << "        Only used if '-m' is specified." << endl;
	cout << " -o=OUTFILE" << endl;
	cout << " --output-file=OUTFILE" << endl;
	cout << "        Write the solution to the file OUTFILE (MatrixMarket format)." << endl;
	cout << " --single-component" << endl;
	cout << "		 Do not break the problem into several components." << endl;
	cout << " -k=METHOD" << endl;
	cout << " --krylov-method=METHOD" << endl;
	cout << "        Specify the iterative Krylov solver:" << endl;
	cout << "        METHOD=0 or METHOD=bicgstab      use BiCGStab" << endl;
	cout << "        METHOD=1 or METHOD=bicgstab2     use BiCGStab(2). This is the default." << endl;
	cout << " --safe-fact" << endl;
	cout << "        Use safe LU-UL factorization." << endl; 
	cout << " --const-band" << endl;
	cout << "        Do not use various-bandwidth-method to solve the problem." << endl; 
	cout << " -f=METHOD" << endl;
	cout << " --factorization-method=METHOD" << endl;
	cout << "        Specify the factorization type used to assemble the reduced matrix" << endl;
	cout << "        METHOD=0 or METHOD=lu_ul                for both applying LU and UL." << endl;
	cout << "        METHOD=1 or METHOD=lu_lu                for applying a complete LU. This is the default." << endl;
	cout << " --precond-method=METHOD" << endl;
	cout << "        Specify the preconditioner to be used" << endl;
	cout << "        METHOD=0 or METHOD=SPIKE                for using SPIKE preconditioner.  This is the default." << endl;
	cout << "        METHOD=1 or METHOD=BLOCK                for using Block preconditionera." << endl;
	cout << " -? -h --help" << endl;
	cout << "        Print this message and exit." << endl;
	cout << endl;
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
	spike::SolverStats stats = mySolver.getStats();

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
	cout << "SPMV count = " << mySpmv.getCount() 
		 << "  total time = " << mySpmv.getTime() 
		 << "  GFlop/s = " << mySpmv.getGFlops()
		 << endl;
	cout << endl;
}
