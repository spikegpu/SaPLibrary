#include <algorithm>
#include <fstream>
#include <cmath>

#include <sap/solver.h>
#include <sap/banded_matrix.h>
#include <sap/spmv.h>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif

// -----------------------------------------------------------------------------
// Macro to obtain a random number between two specified values
// -----------------------------------------------------------------------------
#define RAND(L,H)  ((L) + ((H)-(L)) * (float)rand()/(float)RAND_MAX)


// -----------------------------------------------------------------------------
// Typedefs
// -----------------------------------------------------------------------------
typedef double REAL;
typedef double PREC_REAL;

typedef typename cusp::array1d<REAL, cusp::device_memory>         Vector;
typedef typename sap::BandedMatrix<Vector>                        Matrix;

typedef typename sap::Solver<Vector, PREC_REAL>                 SaPSolver;
typedef typename sap::MVBanded<Matrix>                          SpmvFunctor;

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
      OPT_RTOL, OPT_ATOL, OPT_MAXIT,
      OPT_DROPOFF_FRAC, 
      OPT_BAND, OPT_USE_BCR,
      OPT_OUTFILE, OPT_PRECOND,
      OPT_KRYLOV, OPT_SAFE_FACT,
      OPT_GPU_COUNT};

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
	{ OPT_BAND,          "-b",                   SO_MULTI   },
	{ OPT_BAND,          "--banded-synthetic",   SO_MULTI   },
	{ OPT_OUTFILE,       "-o",                   SO_REQ_CMB },
	{ OPT_OUTFILE,       "--output-file",        SO_REQ_CMB },
	{ OPT_PRECOND,       "--precond-method",     SO_REQ_CMB },
	{ OPT_KRYLOV,        "-k",                   SO_REQ_CMB },
	{ OPT_KRYLOV,        "--krylov-method",      SO_REQ_CMB },
	{ OPT_GPU_COUNT,     "--gpu-count",          SO_REQ_CMB },
	{ OPT_SAFE_FACT,     "--safe-fact",          SO_NONE    },
    { OPT_USE_BCR,       "--use-bcr",            SO_NONE    },
	{ OPT_VERBOSE,       "-v",                   SO_NONE    },
	{ OPT_VERBOSE,       "--verbose",            SO_NONE    },
	{ OPT_HELP,          "-?",                   SO_NONE    },
	{ OPT_HELP,          "-h",                   SO_NONE    },
	{ OPT_HELP,          "--help",               SO_NONE    },
	SO_END_OF_OPTIONS
};

// Color to print
enum TestColor {COLOR_NO = 0,
                COLOR_RED,
                COLOR_GREEN} ;

// -----------------------------------------------------------------------------
// Forward declarations.
// -----------------------------------------------------------------------------
void ShowUsage();
void sapSetDevice();
bool GetProblemSpecs(int             argc, 
                     char**          argv,
                     int&            N,
                     int&            k,
                     REAL&           d,
                     string&         fileSol,
                     int&            numPart,
                     bool&           verbose,
                     sap::Options& opts);

void GetRhsVector(const Matrix& A, Vector& b, Vector& x_target);
void PrintStats(bool               success,
                const SaPSolver& mySolver,
                const SpmvFunctor& mySpmv);

class OutputItem
{
public:
	OutputItem(std::ostream &o): m_o(o), m_additional_item_count(10) {}

	int           m_additional_item_count;

	template <typename T>
	void operator() (T item, TestColor c = COLOR_NO) {
		m_o << "<td style=\"border-style: inset;\">\n";
		switch (c)
		{
			case COLOR_RED:
				m_o << "<p> <FONT COLOR=\"Red\">" << item << " </FONT> </p>\n";
				break;

			case COLOR_GREEN:
				m_o << "<p> <FONT COLOR=\"Green\">" << item << " </FONT> </p>\n";
				break;

			default:
				m_o << "<p> " << item << " </p>\n";
				break;
		}
		m_o << "</td>\n";
	}
private:
	std::ostream &m_o;
};


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
	sap::Options opts;

	opts.trackReordering = false;
	opts.variableBandwidth = false;
	opts.factMethod = sap::LU_UL;
	opts.performReorder = false;
	opts.applyScaling = false;

	if (!GetProblemSpecs(argc, argv, pN, pk, pd, fileSol, numPart, verbose, opts))
		return 1;

	// Get the device with most available memory.
	sapSetDevice();

	// Get matrix and rhs.
	Matrix A(pN, pk, pd);
	Vector b;
	Vector x_target;
	Vector delta_x_target;

	GetRhsVector(A, b, x_target);

	// Create the SAP Solver object and the SPMV functor. Perform the solver
	// setup, then solve the linear system using a 0 initial guess.
	// Set the initial guess to the zero vector.
	SaPSolver  mySolver(numPart, opts);
	SpmvFunctor  mySpmv(A);
	Vector x(A.num_rows, 0);

	bool success = false;

	OutputItem outputItem(cout);
	cout << "<tr valign=top>" << endl;

	outputItem(pN);
	outputItem(pk);
	outputItem(pd);
	outputItem(numPart);

	try {
		mySolver.setup(A);
		success = mySolver.solve(mySpmv, b, x);
	} catch (const std::bad_alloc&) {
		outputItem ("OoM (in setup stage)", COLOR_RED);

		for (int i = 0; i < outputItem.m_additional_item_count; i++)
			outputItem("");

		cout << "</tr>" << endl;

		return 1;
	}

	sap::Stats stats = mySolver.getStats();

	// Reason why cannot solve (for unsuccessful solving only)
	if (success) {
		outputItem ( "OK");
    } else {
		outputItem ( "NConv", COLOR_RED);
    }

	// Calculate the actual residual and its norm.
    {
		Vector r(A.num_rows);
		mySpmv(x, r);
		cusp::blas::axpby(b, r, r, REAL(1.0), REAL(-1.0));

        outputItem( cusp::blas::nrm2(r) / cusp::blas::nrm2(b));
    }

    if (opts.useBCR) {
        outputItem( stats.time_bcr_lu);
        outputItem( stats.time_bcr_sweep_deflation);
        outputItem( stats.time_bcr_mat_mul_deflation);
    } else {
        outputItem( stats.time_offDiags);
        outputItem( stats.time_bandLU + stats.time_bandUL);
        outputItem( stats.time_assembly);
        outputItem( stats.time_fullLU);
    }

	// Total time for setup
	outputItem( stats.timeSetup);
	// Number of iterations to converge
	outputItem( stats.numIterations);

    if (opts.useBCR) {
        outputItem(stats.time_bcr_sweep_inflation);
        outputItem(stats.time_bcr_mv_inflation);
    }

    // Total time for Krylov solve
	outputItem( stats.timeSolve);
    // Total time for Krylov solve per iteration
    outputItem( stats.timeSolve / std::max(1.0f, stats.numIterations));
	// Total amount of time
	outputItem( stats.timeSetup + stats.timeSolve);

	cout << "</tr>" << endl;

	// Write solution file and print solver statistics.
	if (fileSol.length() > 0)
		cusp::io::write_matrix_market_file(x, fileSol);

	/*
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
		sap::Stats stats = mySolver.getStats();
		printf("%d,%d,%d,%g,%g\n", success, pN, pk, pd, stats.timeSetup + stats.timeSolve);
	}*/

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
	int     N = A.num_rows;
	b.resize(N, (REAL)1.0);
	// Create a desired solution vector (on the host), then copy it
	// to the device.
	/*
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
	cusp::multiply(A, x_target, b);  */
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
                sap::Options& opts)
{
	numPart = -1;
	verbose = false;

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
            case OPT_USE_BCR:
                opts.useBCR = true;
                break;
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
			case OPT_GPU_COUNT:
                opts.gpuCount = atoi(args.OptionArg());
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
		opts.factMethod = sap::LU_only;

	// Print out the problem specifications.
	if (verbose) {
		cout << endl;
		cout << "Problem size: " << N << " "<<k <<" "<<d<<endl;
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
			cout << "Factorization method: LU - UL" << endl;
			if (opts.dropOffFraction > 0)
				cout << "Drop-off fraction: " << opts.dropOffFraction << endl;
			else
				cout << "No drop-off." << endl;
			cout << (opts.safeFactorization ? "Use safe factorization." : "Use non-safe fast factorization.") << endl;
		}
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
	cout << " -b SIZE BW DD" << endl;
	cout << " --banded-synthetic SIZE BW DD" << endl;
	cout << "        Use a synthetic banded matrix of size SIZE, half-bandwidth BW," << endl;
	cout << "        and degree of diagonal dominance DD." << endl;
	cout << " -o=OUTFILE" << endl;
	cout << " --output-file=OUTFILE" << endl;
	cout << "        Write the solution to the file OUTFILE (MatrixMarket format)." << endl;
	cout << " -p=NUM_PARTITIONS" << endl;
	cout << " --num-partitions=NUM_PARTITIONS" << endl;
	cout << "        Specify the number of partitions (default 1)." << endl;
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
	cout << "        Frobenius norm is ignored (default 0.0 -- i.e. no drop-off)." << endl;
	cout << " --safe-fact" << endl;
	cout << "        Use safe LU-UL factorization." << endl; 
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
