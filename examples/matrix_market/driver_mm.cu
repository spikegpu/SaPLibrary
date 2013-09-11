#include <algorithm>
#include <fstream>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/blas.h>

#include <spike/solver.h>



// Macro to obtain a random number between two specified values
#define RAND(L,H)  ((L) + ((H)-(L)) * (float)rand()/(float)RAND_MAX)
#define MAX(A,B)   (((A) > (B)) ? (A) : (B))
#define MIN(A,B)   (((A) < (B)) ? (A) : (B))


// -------------------------------------------------------------------
// Typedefs
// -------------------------------------------------------------------
typedef double REAL;

typedef cusp::device_memory MEMORY;
typedef typename cusp::csr_matrix<int, REAL, MEMORY> Matrix;
typedef typename cusp::array1d<REAL, MEMORY>         Vector;

typedef typename cusp::coo_matrix<int, REAL, cusp::host_memory>  MatrixHost;
typedef typename cusp::array1d<REAL, cusp::host_memory>          VectorHost;

typedef typename spike::Solver<Matrix, Vector>       SpikeSolver;
typedef typename spike::SpmvCusp<Matrix, Vector>     SpmvFunctor;
typedef typename spike::SolverOptions				 SolverOptions;


// -------------------------------------------------------------------
// -------------------------------------------------------------------
using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::vector;


// -------------------------------------------------------------------
// Definitions for SimpleOpt and SimpleGlob
// -------------------------------------------------------------------
#include <SimpleOpt/SimpleOpt.h>


// ID values to identify command line arguments
enum {OPT_HELP, OPT_VERBOSE, OPT_PART,
	  OPT_NO_REORDERING, OPT_NO_SCALING,
	  OPT_TOL, OPT_MAXIT,
	  OPT_DROPOFF_FRAC, 
	  OPT_MATFILE, OPT_MATLIST, OPT_RHSFILE, 
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
	{ OPT_MATLIST,		 "--matrix-list",		 SO_REQ_CMB },
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
	{ OPT_VERBOSE,       "-v",                   SO_NONE    },
	{ OPT_VERBOSE,       "--verbose",            SO_NONE    },
    { OPT_HELP,          "-?",                   SO_NONE    },
	{ OPT_HELP,          "-h",                   SO_NONE    },
    { OPT_HELP,          "--help",               SO_NONE    },
	SO_END_OF_OPTIONS
};


// -------------------------------------------------------------------
// Problem types
// Problem definition
// -------------------------------------------------------------------
struct Problem {
	int           N;
	int			  nnz;
	int           k;
	REAL          d;

	int           numPart;

	int           maxIt;
	REAL          tol;
	REAL          fraction;

	bool          reorder;
	bool          scale;

	string        fileMat;
	string        fileRhs;
	string        fileSol;
	string		  fileMatList;

	spike::SolverType    krylov;
	spike::SolverMethod  factorization;
	spike::PrecondMethod precondMethod;

	bool		  singleComponent;
	bool          safeFactorization;
	bool		  variousBandwidth;
	bool		  trackReordering;

	bool          verbose;
};


// -------------------------------------------------------------------
// Forward declarations.
// -------------------------------------------------------------------
void ShowUsage();
static const char* GetLastErrorText(int a_nError) ;

void spikeSetDevice();

bool GetProblemSpecs(int argc, char** argv, Problem& pb);

void GetRhsVector(const Matrix& A, Vector& b, Vector& x_target);

void PrintProblem(const Problem& pb, bool verbose);

void PrintStats(const Problem&	   pb,
				bool               success,
                const SpikeSolver& mySolver,
                const SpmvFunctor& mySpmv,
                bool               verbose);

void ClearStats(const SpikeSolver& mySolver);

bool getNextNonEmptyLine(std::ifstream &fin, string &str);


// -------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------
int main(int argc, char** argv) 
{
	Problem pb;

	pb.N = 0;
	pb.nnz = 0;
	pb.k = 0;
	pb.d = 1.0;
	pb.maxIt = 100;
	pb.tol = 1e-6;
	pb.fraction = 0.0;
	pb.reorder = true;
	pb.scale = true;
	pb.numPart = 1;

	pb.factorization = spike::LU_only;
	pb.precondMethod = spike::Spike;
	pb.singleComponent = false;
	pb.safeFactorization = false;
	pb.variousBandwidth = true;
	pb.krylov = spike::BiCGStab2;
	pb.trackReordering = true;

	pb.verbose = false;


	// Get the problem specification from the program arguments.
	if (!GetProblemSpecs(argc, argv, pb))
		return 1;

	if (!pb.reorder)
		pb.variousBandwidth = false;

	if (pb.variousBandwidth)
		pb.factorization = spike::LU_only;

	if (pb.fileMat.length() > 0)
		pb.trackReordering = false;
	else
		pb.trackReordering = true;

	// Print information on the problem that will be solved.
	if (pb.verbose)
		PrintProblem(pb, pb.verbose);


	// Get the device with most available memory.
	spikeSetDevice();


	// Get matrix and rhs. Note that the 'target' solution is only
	// set if using a generated RHS.
	Matrix A;
	Vector b;
	Vector x_target;
	Vector delta_x_target;

	std::ifstream fin;

	if (pb.fileMat.length() > 0)
		cusp::io::read_matrix_market_file(A, pb.fileMat);
	else {
		fin.open(pb.fileMatList.data(), std::ios::in);
		if (!getNextNonEmptyLine(fin, pb.fileMat)) {
			fprintf(stderr, "The matrix list file %s cannot be opened or there is no matrix file specified at all.\n", pb.fileMatList.data());
			exit(-1);
		}
		cusp::io::read_matrix_market_file(A, pb.fileMat);
	}

	pb.N = A.num_rows;
	pb.nnz = A.num_entries;
	if (pb.fileRhs.length() > 0)
		cusp::io::read_matrix_market_file(b, pb.fileRhs);
	else
		GetRhsVector(A, b, x_target);

	SolverOptions solverOptions;
	solverOptions.numPartitions = pb.numPart;

	solverOptions.solverType = pb.krylov;
	solverOptions.maxNumIterations = pb.maxIt;
	solverOptions.tolerance = pb.tol;

	solverOptions.performReorder = pb.reorder;
	solverOptions.applyScaling = pb.scale;
	solverOptions.dropOffFraction = pb.fraction;

	solverOptions.method = pb.factorization;
	solverOptions.precondMethod = pb.precondMethod;
	solverOptions.safeFactorization = pb.safeFactorization;
	solverOptions.variousBandwidth = pb.variousBandwidth;
	solverOptions.trackReordering = pb.trackReordering;

	// Create the SPIKE Solver object and the SPMV functor.
	// Set the initial guess to the zero vector.
	// SpikeSolver  mySolver(pb.numPart, pb.maxIt, pb.tol, pb.reorder, pb.scale, pb.fraction, pb.krylov, pb.factorization, pb.precondMethod, pb.singleComponent, pb.safeFactorization, pb.variousBandwidth, pb.trackReordering);
	SpikeSolver mySolver(solverOptions);

	for (int i=0; ; i++) {

		if (i > 0) {
			if (!getNextNonEmptyLine(fin, pb.fileMat))
				break;
			cusp::io::read_matrix_market_file(A, pb.fileMat);
		}

		SpmvFunctor  mySpmv(A);
		Vector       x(pb.N, 0);

		if (i == 0)
			mySolver.setup(A);
		else
			mySolver.update(A.values);

		bool success = mySolver.solve(mySpmv, b, x);

		// If an output file was specified, write the solution vector
		// in MatrixMarket format.
		if (pb.fileSol.length() > 0)
			cusp::io::write_matrix_market_file(x, pb.fileSol);

		// Print solution statistics.
		PrintStats(pb, success, mySolver, mySpmv, pb.verbose);

		// Calculate the actual residual and its norm.
		if (pb.verbose) {
			Vector r(pb.N);
			mySpmv(x, r);
			cusp::blas::axpby(b, r, r, REAL(1.0), REAL(-1.0));
			cout << "|b - A*x|      = " << cusp::blas::nrm2(r) << endl;
			cout << "|b|            = " << cusp::blas::nrm2(b) << endl;	

			// If we have used a generated RHS, print the difference
			// between the target solution and the obtained solution.
			//    x_target <- x_target - x
			if (pb.fileRhs.length() == 0) {
				cout << "|x_target|     = " << cusp::blas::nrm2(x_target) << endl;
				delta_x_target.resize(pb.N);
				cusp::blas::axpby(x_target, x, delta_x_target, REAL(1.0), REAL(-1.0));
				cout << "|x_target - x| = " << cusp::blas::nrm2(delta_x_target) << endl;
			}
		}

		ClearStats(mySolver);

		if (pb.fileMat.length() == 0)
			break;
	}

	if (fin.is_open())
		fin.close();

	// That's all folks!
	return 0;
}

// -------------------------------------------------------------------
// spikeSetDevice()
//
// This function gets the device with maximum free space and set that
// device to working device. 
// FIXME:
// Note that this function shall be removed when we start multi-gpu
// support. 
// -------------------------------------------------------------------
void spikeSetDevice() {
	int deviceCount = 0;
	
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		cudaSetDevice(0);
		return;
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

	// fprintf(stderr, "Use Device: %d\n", max_idx);
	std::cerr << "Use device: " << max_idx << std::endl;
	cudaSetDevice(max_idx);
}

// -------------------------------------------------------------------
// GetProblemSpecs()
//
// This function parses the specified program arguments and sets up
// the problem to be solved.
// -------------------------------------------------------------------
bool
GetProblemSpecs(int argc, char** argv, Problem& pb)
{
	// Create the option parser and pass it the arguments from main
	// and the array of valid options.
	CSimpleOptA args(argc, argv, g_options);

	// Loop while there are arguments to be processed.
	while (args.Next()) {
		// Exit immediately if we encounter an invalid argument.
		if (args.LastError() != SO_SUCCESS) {
			cout << "Invalid argument: " << args.OptionText() << endl;
			ShowUsage();
			return false;
		}

		switch (args.OptionId()) {

		case OPT_HELP:
			ShowUsage();
			return false;

		case OPT_PART:
			pb.numPart = atoi(args.OptionArg());
			if (pb.numPart <= 0) {
				cout << "Invalid value for the number of partitions. P = " << pb.numPart << endl;
				return false;
			}
			break;

		case OPT_TOL:
			pb.tol = atof(args.OptionArg());
			break;
			
		case OPT_MAXIT:
			pb.maxIt = atoi(args.OptionArg());
			break;

		case OPT_DROPOFF_FRAC:
			pb.fraction = atof(args.OptionArg());
			break;

		case OPT_NO_REORDERING:
			pb.reorder = false;
			break;

		case OPT_NO_SCALING:
			pb.scale = false;
			break;

		case OPT_VERBOSE:
			pb.verbose = true;
			break;

		case OPT_MATFILE:
			pb.fileMat = args.OptionArg();
			break;

		case OPT_MATLIST:
			pb.fileMatList = args.OptionArg();
			break;

		case OPT_RHSFILE:
			pb.fileRhs = args.OptionArg();
			break;

		case OPT_OUTFILE:
			pb.fileSol = args.OptionArg();
			break;

		case OPT_FACTORIZATION:
			{
				string fact = args.OptionArg();
				std::transform(fact.begin(), fact.end(), fact.begin(), ::toupper);
				if (fact == "0" || fact == "LU_UL")
					pb.factorization = spike::LU_UL;
				else if (fact == "1" || fact == "LU_LU")
					pb.factorization = spike::LU_only;
				else
					return false;
			}

			break;

		case OPT_PRECOND:
			{
				string precond = args.OptionArg();
				std::transform(precond.begin(), precond.end(), precond.begin(), ::toupper);
				if (precond == "0" || precond == "SPIKE")
					pb.precondMethod = spike::Spike;
				else if(precond == "1" || precond == "BLOCK")
					pb.precondMethod = spike::Block;
				else
					return false;
			}

			break;

		case OPT_KRYLOV:
			{
				string kry = args.OptionArg();
				std::transform(kry.begin(), kry.end(), kry.begin(), ::toupper);
				if (kry == "0" || kry == "BICGSTAB")
					pb.krylov = spike::BiCGStab;
				else if (kry == "1" || kry == "BICGSTAB2")
					pb.krylov = spike::BiCGStab2;
				else
					return false;
			}

			break;

		case OPT_SINGLE_COMP:
			pb.singleComponent = true;
			break;

		case OPT_SAFE_FACT:
			pb.safeFactorization = true;
			break;

		case OPT_CONST_BAND:
			pb.variousBandwidth = false;
			break;
		}

	}

	// If no problem was defined, show usage and exit.
	if (pb.fileMat.length() == 0 && pb.fileMatList.length() == 0) {
		cout << "No matrix file was defined!" << endl << endl;
		ShowUsage();
		return false;
	}

	return true;
}


// -------------------------------------------------------------------
// ShowUsage()
//
// This function displays the correct usage of this program
// -------------------------------------------------------------------
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
	cout << " --matrix-list=MATLISTFILE" << endl;
	cout << "        Specify a list of matrices (MatrixMarket format) in the file MATLISTFILE." << endl;
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
	cout << " -v --verbose" << endl;
	cout << "        Verbose output." << endl; 
	cout << " -? -h --help" << endl;
	cout << "        Print this message and exit." << endl;
	cout << endl;
}


// -------------------------------------------------------------------
// GetLastErrorText()
//
// This function translates SO error codes to human readable strings.
// -------------------------------------------------------------------
static const char* GetLastErrorText(int a_nError) 
{
    switch (a_nError) {
    case SO_SUCCESS:            return "Success";
    case SO_OPT_INVALID:        return "Unrecognized option";
    case SO_OPT_MULTIPLE:       return "Option matched multiple strings";
    case SO_ARG_INVALID:        return "Option does not accept argument";
    case SO_ARG_INVALID_TYPE:   return "Invalid argument format";
    case SO_ARG_MISSING:        return "Required argument is missing";
    case SO_ARG_INVALID_DATA:   return "Invalid argument data";
    default:                    return "Unknown error";
    }
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

	VectorHost xh(N);

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


// -------------------------------------------------------------------
// PrintProblem()
//
// This function prints information about the problem and solution
// parameters.
// -------------------------------------------------------------------
void PrintProblem(const Problem& pb, bool verbose)
{
	string factTypeStr;
	string precondStr;
	switch (pb.factorization) {
	case spike::LU_UL:
		factTypeStr = "LU-UL";
		break;
	case spike::LU_only:
		factTypeStr = "LU-LU";
		break;
	}

	switch (pb.precondMethod) {
	case spike::Spike:
		precondStr = "SPIKE";
		break;
	case spike::Block:
		precondStr = "BLOCK";
		break;
	}

	string krylovTypeStr;
	switch (pb.krylov) {
	case spike::BiCGStab:
		krylovTypeStr = "BiCGStab";
		break;
	case spike::BiCGStab2:
		krylovTypeStr = "BiCGStab2";
		break;
	}

	if (verbose) {
		cout << endl;
		cout << "Application problem from file" << endl
			<< "  Matrix file: " << pb.fileMat << endl;
		if (pb.fileRhs.length() > 0)
			cout << "  Rhs file: " << pb.fileRhs << endl;
		cout << "Using " << pb.numPart << " partition";
		if (pb.numPart > 1)
			cout << "s";
		cout << "." << endl;
		cout << "Iterative solver: " << krylovTypeStr << endl;
		cout << "Tolerance: " << pb.tol << endl;
		cout << "Max. iterations: " << pb.maxIt << endl;
		if (pb.fraction > 0)
			cout << "Drop-off fraction: " << pb.fraction << endl;
		else
			cout << "No drop-off." << endl;
		cout << (pb.singleComponent ? "Do not break the problem into several components." : "Attempt to break the problem into several components.") << endl;
		cout << (pb.reorder ? "Perform reordering." : "Do not perform reordering.") << endl;
		cout << (pb.scale   ? "Apply scaling." : "Do not apply scaling.") << endl;
		cout << (pb.safeFactorization ? "Using safe factorization." : "Using non-safe fast factorization.") << endl;
		cout << (pb.variousBandwidth ? "Using various-bandwidth method." : "Not using various-bandwidth method.") << endl;
		cout << "Factorization method: " << factTypeStr << endl;
		cout << "Preconditioner: " << precondStr << endl;
		if (pb.fileRhs.length() > 0)
			cout << "Sol file: " << pb.fileSol << endl;
		cout << endl << endl;
	} else {
		cout << pb.fileMat;
		cout << "     " << pb.numPart;
		cout << "     " << pb.maxIt << "  " << pb.tol;
		cout << "     " << (pb.reorder ? "T" : "F") << (pb.scale ? "T" : "F");
		cout << "     " << pb.fraction;
		cout << "     " << factTypeStr << "  " << precondStr << "   "<<(pb.safeFactorization ? "T" : "F");
		cout << "	  " << (pb.variousBandwidth ? "T" : "F");
		cout << endl;
	}
}


// -------------------------------------------------------------------
// PrintStats()
//
// This function prints solver statistics.
// -------------------------------------------------------------------
void PrintStats(const Problem&	   pb,
				bool               success,
                const SpikeSolver& mySolver,
                const SpmvFunctor& mySpmv,
                bool               verbose)
{
	spike::SolverStats stats = mySolver.getStats();

	if (verbose) {
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
	} else {
		string matName = pb.fileMat;
		unsigned end_pos = matName.rfind(".mtx");
		if (end_pos == string::npos)
			end_pos = matName.size();
		unsigned start_pos;
		for (start_pos = end_pos - 1; start_pos > 0; start_pos --)
			if (matName[start_pos] == '/' || matName[start_pos] == '\\') {
				start_pos++;
				break;
			}
		matName = matName.substr(start_pos, end_pos - start_pos);
		cout << matName				   << ",";
		cout << pb.N				   << ",";
		cout << pb.nnz				   << ",";
		cout << stats.bandwidth		   << ",";
		cout << (success ? "T" : "F")  << ",";
		cout << pb.numPart			   << ",";
		cout << stats.numIterations    << ",";
		cout << stats.time_reorder	   << ",";
		cout << stats.time_cpu_assemble<< ",";
		cout << stats.time_transfer	   << ",";
		cout << stats.time_offDiags	   << ",";
		cout << stats.time_bandLU	   << ",";
		cout << stats.time_bandUL      << ",";
		cout << stats.time_assembly	   << ",";
		cout << stats.time_fullLU	   << ",";
		cout << stats.timeSetup        << ",";
		cout << stats.timeSolve        << ",";
		cout << stats.timeSetup
				+ stats.timeSolve;
		cout << endl;
	}
}

// -------------------------------------------------------------------
// ClearStats()
//
// This function clears solver statistics.
// -------------------------------------------------------------------
void ClearStats(const SpikeSolver& mySolver)
{
	spike::SolverStats stats = mySolver.getStats();

	stats.numIterations = 0;
	stats.residualNorm = 0;
	stats.bandwidthReorder = 0;
	stats.bandwidth = 0;
	stats.actualDropOff = 0;

	stats.timeSetup = 0;
	stats.time_reorder = 0;
	stats.time_cpu_assemble = 0;
	stats.time_transfer = 0;
	stats.time_toBanded = 0;
	stats.time_offDiags = 0;
	stats.time_bandLU = 0;
	stats.time_bandUL = 0;
	stats.time_assembly = 0;
	stats.time_fullLU = 0;
	stats.timeSolve = 0;
	stats.time_shuffle = 0;
}

bool getNextNonEmptyLine(std::ifstream &fin, string &str)
{
	if (!fin.is_open())
		return false;

	while (getline(fin, str)) {
		int length = str.length();

		if (length == 0)
			continue;
		
		for (int i=0; i < length; i++)
			if (str[i] > ' ')
				return true;
	}
	return false;
}
