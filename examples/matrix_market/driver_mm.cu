#include <algorithm>
#include <fstream>

#include <sap/solver.h>
#include <sap/spmv.h>
#include <sap/exception.h>

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
typedef typename sap::SpmvCusp<Matrix>                          SpmvFunctor;


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
	  OPT_SPD,
      OPT_NO_REORDERING, OPT_NO_DB, OPT_NO_SCALING,
      OPT_RTOL, OPT_ATOL, OPT_MAXIT,
      OPT_DROPOFF_FRAC, OPT_MAX_BANDWIDTH,
      OPT_MATFILE, OPT_RHSFILE,
      OPT_OUTFILE, OPT_FACTORIZATION, OPT_PRECOND,
      OPT_KRYLOV, OPT_SAFE_FACT,
      OPT_CONST_BAND};

// Table of CSimpleOpt::Soption structures. Each entry specifies:
// - the ID for the option (returned from OptionId() during processing)
// - the option as it should appear on the command line
// - type of the option
// The last entry must be SO_END_OF_OPTIONS
CSimpleOptA::SOption g_options[] = {
	{ OPT_PART,          "-p",                   SO_REQ_CMB },
	{ OPT_PART,          "--num-partitions",     SO_REQ_CMB },
	{ OPT_RTOL,          "-t",                   SO_REQ_CMB },
	{ OPT_RTOL,          "--tolerance",          SO_REQ_CMB },
	{ OPT_RTOL,          "--relTol",             SO_REQ_CMB },
	{ OPT_ATOL,          "--absTol",             SO_REQ_CMB },
	{ OPT_MAXIT,         "-i",                   SO_REQ_CMB },
	{ OPT_MAXIT,         "--max-num-iterations", SO_REQ_CMB },
	{ OPT_DROPOFF_FRAC,  "-d",                   SO_REQ_CMB },
	{ OPT_DROPOFF_FRAC,  "--drop-off-fraction",  SO_REQ_CMB },
	{ OPT_MAX_BANDWIDTH, "-b",                   SO_REQ_CMB },
	{ OPT_MAX_BANDWIDTH, "--max-bandwidth",      SO_REQ_CMB },
	{ OPT_MATFILE,       "-m",                   SO_REQ_CMB },
	{ OPT_MATFILE,       "--matrix-file",        SO_REQ_CMB },
	{ OPT_RHSFILE,       "-r",                   SO_REQ_CMB },
	{ OPT_RHSFILE,       "--rhs-file",           SO_REQ_CMB },
	{ OPT_OUTFILE,       "-o",                   SO_REQ_CMB },
	{ OPT_OUTFILE,       "--output-file",        SO_REQ_CMB },
	{ OPT_NO_REORDERING, "--no-reordering",      SO_NONE    },
	{ OPT_NO_DB,         "--no-db",              SO_NONE    },
	{ OPT_NO_SCALING,    "--no-scaling",         SO_NONE    },
	{ OPT_FACTORIZATION, "-f",                   SO_REQ_CMB },
	{ OPT_FACTORIZATION, "--factorization-method", SO_REQ_CMB },
	{ OPT_PRECOND,       "--precond-method",     SO_REQ_CMB },
	{ OPT_KRYLOV,        "-k",                   SO_REQ_CMB },
	{ OPT_KRYLOV,        "--krylov-method",      SO_REQ_CMB },
	{ OPT_SAFE_FACT,     "--safe-fact",          SO_NONE    },
	{ OPT_CONST_BAND,    "--const-band",         SO_NONE    },
	{ OPT_HELP,          "-?",                   SO_NONE    },
	{ OPT_HELP,          "-h",                   SO_NONE    },
	{ OPT_HELP,          "--help",               SO_NONE    },
	{ OPT_SPD,           "--spd",                SO_NONE    },
	SO_END_OF_OPTIONS
};


// -----------------------------------------------------------------------------
// Forward declarations.
// -----------------------------------------------------------------------------
void ShowUsage();
void sapSetDevice();
bool GetProblemSpecs(int             argc, 
                     char**          argv,
                     string&         fileMat,
                     string&         fileRhs,
                     string&         fileSol,
                     int&            numPart,
                     sap::Options& opts);
void PrintStats(bool               success,
                const SaPSolver& mySolver,
                const SpmvFunctor& mySpmv);


// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) 
{
	// Set up the problem to be solved.
	string         fileMat;
	string         fileRhs;
	string         fileSol;
	int            numPart;
	sap::Options opts;

	if (!GetProblemSpecs(argc, argv, fileMat, fileRhs, fileSol, numPart, opts))
		return 1;

	// Get the device with most available memory.
	sapSetDevice();

	// Get matrix and rhs.
	Matrix A;
	Vector b;

	cusp::io::read_matrix_market_file(A, fileMat);

	if (fileRhs.length() > 0)
		cusp::io::read_matrix_market_file(b, fileRhs);
	else
		b.resize(A.num_rows, 1);

	// Create the SAP Solver object and the SPMV functor. Perform the solver
	// setup, then solve the linear system using a 0 initial guess.
	// Set the initial guess to the zero vector.
	SaPSolver  mySolver(numPart, opts);
	SpmvFunctor  mySpmv(A);
	Vector x(A.num_rows, 0);
	bool   success;

	try {
		mySolver.setup(A);
		success = mySolver.solve(mySpmv, b, x);
	} catch (const std::bad_alloc& e) {
		std::cout << "Exception (bad_alloc): " << e.what() << std::endl;
		return 1;
	} catch (const sap::system_error& e) {
		std::cout << "Exception (system_error): " << e.what() << " Error code: " << e.reason() << std::endl;
		return 1;
	}


	// Write solution file and print solver statistics.
	if (fileSol.length() > 0)
		cusp::io::write_matrix_market_file(x, fileSol);

	PrintStats(success, mySolver, mySpmv);

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
                string&         fileRhs,
                string&         fileSol,
                int&            numPart,
                sap::Options& opts)
{
	numPart = -1;

	// Create the option parser and pass it the program arguments and the array
	// of valid options. Then loop for as long as there are arguments to be
	// processed.
	CSimpleOptA args(argc, argv, g_options);

	bool  maxBandwidth_specified = false;

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
			case OPT_RTOL:
				opts.relTol = atof(args.OptionArg());
				break;
			case OPT_ATOL:
				opts.absTol = atof(args.OptionArg());
				break;
			case OPT_MAXIT:
				opts.maxNumIterations = atoi(args.OptionArg());
				break;
			case OPT_DROPOFF_FRAC:
				opts.dropOffFraction = atof(args.OptionArg());
				break;
			case OPT_MAX_BANDWIDTH:
				opts.maxBandwidth = atoi(args.OptionArg());
				maxBandwidth_specified = true;
				break;
			case OPT_NO_REORDERING:
				opts.performReorder = false;
				break;
			case OPT_NO_DB:
				opts.performDB = false;
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
						opts.factMethod = sap::LU_UL;
					else if (fact == "1" || fact == "LU_LU")
						opts.factMethod = sap::LU_only;
					else
						return false;
				}
				break;
			case OPT_PRECOND:
				{
					string precond = args.OptionArg();
					std::transform(precond.begin(), precond.end(), precond.begin(), ::toupper);
					if (precond == "0" || precond == "SPIKE")
						opts.precondType = sap::Spike;
					else if(precond == "1" || precond == "BLOCK")
						opts.precondType = sap::Block;
					else if(precond == "2" || precond == "NONE")
						opts.precondType = sap::None;
					else
						return false;
				}
				break;
			case OPT_KRYLOV:
				{
					string kry = args.OptionArg();
					std::transform(kry.begin(), kry.end(), kry.begin(), ::toupper);
					if (kry == "0" || kry == "BICGSTAB_C")
						opts.solverType = sap::BiCGStab_C;
					else if (kry == "1" || kry == "GMRES_C")
						opts.solverType = sap::GMRES_C;
					else if (kry == "2" || kry == "CG_C")
						opts.solverType = sap::CG_C;
					else if (kry == "3" || kry == "CR_C")
						opts.solverType = sap::CR_C;
					else if (kry == "4" || kry == "BICGSTAB1")
						opts.solverType = sap::BiCGStab1;
					else if (kry == "5" || kry == "BICGSTAB2")
						opts.solverType = sap::BiCGStab2;
					else if (kry == "6" || kry == "BICGSTAB")
						opts.solverType = sap::BiCGStab;
					else if (kry == "7" || kry == "MINRES")
						opts.solverType = sap::MINRES;
					else
						return false;
				}
				break;
			case OPT_SAFE_FACT:
				opts.safeFactorization = true;
				break;
			case OPT_CONST_BAND:
				opts.variableBandwidth = false;
				break;
			case OPT_SPD:
				opts.isSPD   = true;
				opts.saveMem = true;
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
	if (!opts.performReorder) {
		opts.performDB = false;
		opts.variableBandwidth = false;
	}

	// If no DB reordering, force no scaling.
	if (!opts.performDB)
		opts.applyScaling = false;

	// If using variable bandwidth, force using LU factorization.
	if (opts.variableBandwidth)
		opts.factMethod = sap::LU_only;

	// Print out the problem specifications.
	cout << endl;
	cout << "Matrix file: " << fileMat << endl;
	if (fileRhs.length() > 0)
		cout << "Rhs file:    " << fileRhs << endl;
	if (fileSol.length() > 0)
		cout << "Sol file:    " << fileSol << endl;
	cout << "Iterative solver: ";
	switch (opts.solverType) {
		case sap::BiCGStab_C:
			cout << "BiCGStab (Cusp)" << endl; break;
		case sap::GMRES_C:
			cout << "GMRES (Cusp)" << endl; break;
		case sap::CG_C:
			cout << "CG (Cusp)" << endl; break;
		case sap::CR_C:
			cout << "CR (Cusp)" << endl; break;
		case sap::BiCGStab1:
			cout << "BiCGStab1 (SaP::GPU)" << endl; break;
		case sap::BiCGStab2:
			cout << "BiCGStab2 (SaP::GPU)" << endl; break;
		case sap::BiCGStab:
			cout << "BiCGStab (SaP::GPU)" << endl; break;
		case sap::MINRES:
			cout << "MINRES (SaP::GPU)" << endl; break;
	}
	cout << "Relative tolerance: " << opts.relTol << endl;
	cout << "Absolute tolerance: " << opts.absTol << endl;
	cout << "Max. iterations: " << opts.maxNumIterations << endl;
	cout << "Preconditioner: ";
	switch (opts.precondType) {
		case sap::Spike:
			cout << "SPIKE" << endl; break;
		case sap::Block:
			cout << "BLOCK DIAGONAL" << endl; break;
		case sap::None:
			cout << "NONE" << endl; break;
	}
	if (opts.precondType != sap::None) {
		cout << "Using " << numPart << (numPart ==1 ? " partition." : " partitions.") << endl;
		cout << "Factorization method: " << (opts.factMethod == sap::LU_UL ? "LU - UL" : "LU - LU") << endl;
		if (opts.dropOffFraction > 0)
			cout << "Drop-off fraction: " << opts.dropOffFraction << endl;
		else
			cout << "No drop-off." << endl;
		if (maxBandwidth_specified)
			cout << "Maximum bandwidth: " << opts.maxBandwidth << endl;
		cout << (opts.performReorder ? "Perform reordering." : "Do not perform reordering.") << endl;
		cout << (opts.performDB ? "Perform DB reordering." : "Do not perform DB reordering.") << endl;
		cout << (opts.applyScaling ? "Apply scaling." : "Do not apply scaling.") << endl;
		cout << (opts.safeFactorization ? "Use safe factorization." : "Use non-safe fast factorization.") << endl;
		cout << (opts.variableBandwidth ? "Use variable bandwidth method." : "Use constant bandwidth method.") << endl;
	}
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
	cout << "Usage:  driver_mm -p=NUM_PARTITIONS -m=MATFILE [OPTIONS]" << endl;
	cout << endl;
	cout << " -p=NUM_PARTITIONS" << endl;
	cout << " --num-partitions=NUM_PARTITIONS" << endl;
	cout << "        Specify the number of partitions." << endl;
	cout << " --no-reordering" << endl;
	cout << "        Do not perform reordering (default false)." << endl;
	cout << " --no-db" << endl;
	cout << "        Do not perform DB reordering (ignored if --no-reordering is specified; default false)." << endl;
	cout << " --no-scaling" << endl;
	cout << "        Do not perform DB scaling (ignored if --no-reordering or --no-db is specified; default false)." << endl;
	cout << " -t=TOLERANCE" << endl;
	cout << " --tolerance=TOLERANCE" << endl;
	cout << " --relTol=TOLERANCE" << endl;
	cout << "        Use relative tolerance TOLERANCE for Krylov stopping criteria (default 1e-6)." << endl;
	cout << " --absTol=TOLERANCE" << endl;
	cout << "        Use absolute tolerance TOLERANCE for Krylov stopping criteria (default 0)." << endl;
	cout << " -i=ITERATIONS" << endl;
	cout << " --max-num-iterations=ITERATIONS" << endl;
	cout << "        Use at most ITERATIONS for Krylov solver (default 100)." << endl;
	cout << " -d=FRACTION" << endl;
	cout << " --drop-off-fraction=FRACTION" << endl;
	cout << "        Drop off-diagonal elements such that FRACTION of the matrix" << endl;
	cout << "        elementwise norm-1 is ignored (default 0.0 -- i.e. no drop-off)." << endl;
	cout << " -b=MAX_BANDWIDTH" << endl;
	cout << " --max-bandwidth=MAX_BANDWIDTH" << endl;
	cout << "        Drop off elements such that the bandwidth is at most MAX_BANDWIDTH" << endl;
	cout << " -m=MATFILE" << endl;
	cout << " --matrix-file=MATFILE" << endl;
	cout << "        Read the matrix from the file MATFILE (MatrixMarket format)." << endl;
	cout << " -r=RHSFILE" << endl;
	cout << " --rhs-file=RHSFILE" << endl;
	cout << "        Read the right-hand side vector from the file RHSFILE (MatrixMarket format)." << endl;
	cout << "        If not specified, a right-hand side of all ones is used." << endl;
	cout << " -o=OUTFILE" << endl;
	cout << " --output-file=OUTFILE" << endl;
	cout << "        Write the solution to the file OUTFILE (MatrixMarket format)." << endl;
	cout << " -k=METHOD" << endl;
	cout << " --krylov-method=METHOD" << endl;
	cout << "        Specify the iterative Krylov solver:" << endl;
	cout << "        METHOD=0 or METHOD=BICGSTAB_C    use BiCGStab (Cusp)" << endl;
	cout << "        METHOD=1 or METHOD=GMRES_C       use GMRES (Cusp)" << endl;
	cout << "        METHOD=2 or METHOD=CG_C          use CG (Cusp)" << endl;
	cout << "        METHOD=3 or METHOD=CR_C          use CR (Cusp)" << endl;
	cout << "        METHOD=4 or METHOD=BICGSTAB1     use BiCGStab(1) (SaP::GPU)" << endl;
	cout << "        METHOD=5 or METHOD=BICGSTAB2     use BiCGStab(2) (SaP::GPU). This is the default." << endl;
	cout << "        METHOD=6 or METHOD=BICGSTAB      use BiCGStab (SaP::GPU)" << endl;
	cout << "        METHOD=7 or METHOD=MINRES        use MINRES (SaP::GPU)" << endl;
	cout << " --safe-fact" << endl;
	cout << "        Use safe LU-UL factorization." << endl; 
	cout << " --const-band" << endl;
	cout << "        Force using the constant-bandwidth method (default false)." << endl; 
	cout << " -f=METHOD" << endl;
	cout << " --factorization-method=METHOD" << endl;
	cout << "        Specify the factorization type used to assemble the reduced matrix" << endl;
	cout << "        METHOD=0 or METHOD=lu_ul         use LU and UL for right- and left-spikes." << endl;
	cout << "        METHOD=1 or METHOD=lu_lu         use LU for both right- and left-spikes. This is the default." << endl;
	cout << " --precond-method=METHOD" << endl;
	cout << "        Specify the preconditioner to be used" << endl;
	cout << "        METHOD=0 or METHOD=SPIKE         SPIKE preconditioner.  This is the default." << endl;
	cout << "        METHOD=1 or METHOD=BLOCK         Block-diagonal preconditioner." << endl;
	cout << "        METHOD=2 or METHOD=NONE          no preconditioner." << endl;
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
                const SaPSolver& mySolver,
                const SpmvFunctor& mySpmv)
{
	sap::Stats stats = mySolver.getStats();

	cout << endl;
	cout << (success ? "Success" : "Failed") << endl;

	cout << "Code: " << mySolver.getMonitorCode();
	cout << "  " << mySolver.getMonitorMessage() << endl;

	cout << "Number of iterations = " << stats.numIterations << endl;
	cout << "RHS norm             = " << stats.rhsNorm << endl;
	cout << "Residual norm        = " << stats.residualNorm << endl;
	cout << "Rel. residual norm   = " << stats.relResidualNorm << endl;
	cout << endl;
	cout << "Bandwidth after DB         = " << stats.bandwidthDB << endl;
	cout << "Bandwidth after reordering = " << stats.bandwidthReorder << endl;
	cout << "Bandwidth after drop-off   = " << stats.bandwidth << endl;
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
