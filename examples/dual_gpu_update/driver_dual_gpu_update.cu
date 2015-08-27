// -----------------------------------------------------------------------------
// 
// -----------------------------------------------------------------------------
#include <algorithm>

#include <sap/solver.h>
#include <sap/timer.h>

#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif

#include <omp.h>

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
	  OPT_SYSTEM_NUM,
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
	{ OPT_SYSTEM_NUM,    "-n",                   SO_REQ_CMB },
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

bool GetProblemSpecs(int             argc, 
                     char**          argv,
                     string&         fileMat,
                     int&            numPart,
					 int&            systemCount,
                     sap::Options& opts);

void PrintStats(bool                success,
                const sap::Stats&   stats);



// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv) 
{
	// Set up the problem to be solved.
	string         fileMat;
	int            numPart;
	int            systems_to_solve = 20;
	sap::Options   opts;

	if (!GetProblemSpecs(argc, argv, fileMat, numPart, systems_to_solve, opts))
		return 1;

	int            init_systems_to_solve = systems_to_solve;

	opts.trackReordering = true;
	
	// Get matrix and rhs.
	Matrix A[2];
	SaPSolver *mySolver[2];
	CustomSpmv  *mySpmv[2];
	Vector      b[2];
	Vector      x[2];
	cudaSetDevice(0);
	cusp::io::read_matrix_market_file(A[0], fileMat);

	// Create the SAP Solver object and the custom SPMV functor.
	mySolver[0] = new SaPSolver(numPart, opts);
	mySpmv[0]   = new CustomSpmv(A[0]);
	b[0].resize(A[0].num_rows, 1.0);
	x[0].resize(A[0].num_rows);

	cudaSetDevice(1);
	cusp::io::read_matrix_market_file(A[1], fileMat);
	mySolver[1] = new SaPSolver(numPart, opts);
	mySpmv[1]   = new CustomSpmv(A[1]);
	b[1].resize(A[1].num_rows, 1.0);
	x[1].resize(A[1].num_rows);

	sap::CPUTimer loc_timer;
	double elapsed;
	omp_set_num_threads(2);

	int working_thread = 0;  // Specify the thread to solve the problem while
							 // the other thread updates the preconditioner
	omp_lock_t g_lock;
	omp_init_lock(&g_lock);
	
	loc_timer.Start();

	// The preconditioner on the first GPU should be set up now.
	cudaSetDevice(0);
	mySolver[working_thread] -> setup(A[working_thread]);
	bool update_done = false;
	bool solveSuccess = true;
	bool matrix_updated = false;
	bool system_changed = false;
	// Solve the linear system A*x = b for two different RHS.
	// In each case, set the initial guess to 0.
#pragma omp parallel shared(init_systems_to_solve, systems_to_solve, working_thread, g_lock, update_done, solveSuccess, matrix_updated, system_changed, opts)
	{
		int tid = omp_get_thread_num();
		cudaSetDevice(tid);

		while (true) {
			int loc_working_thread;
			omp_set_lock(&g_lock);
			loc_working_thread = working_thread;
			omp_unset_lock(&g_lock);

			if (tid == loc_working_thread) {

				// The working thread should solve the problem
				cusp::blas::fill(x[tid], 0.0);
				solveSuccess = mySolver[tid] -> solve(*(mySpmv[tid]), b[tid], x[tid]);

				// If the problem is successfully solved, do follow-up computations
				// here, and assemble the new system (say, new Jacobian matrix)
				if (solveSuccess) {
					systems_to_solve --;

					// Do computations, and assemble the RHS in the next step
					//// thrust::copy(x[tid].begin(), x[tid].end(), b[tid].begin());

					// Assemble new system
					cusp::blas::scal(A[tid].values, 1.02);

					matrix_updated = true;

				}
			} else if (tid == 1 - loc_working_thread && matrix_updated) {
				bool updateSuccess;

				// An update failure implies the preconditioner has not been
				// set up, so here this thread either does a new set up
				// or an update.
				try {
					updateSuccess = mySolver[tid] -> update(A[tid].values);
				} catch (const sap::system_error &e) {
					updateSuccess = false;
				}

				if (!updateSuccess)
					mySolver[tid] -> setup(A[tid]);

				omp_set_lock(&g_lock);
				working_thread = 1 - working_thread;
				update_done = true;
				omp_unset_lock(&g_lock);
			}

			// If the preconditioner has been updated, or the working thread fails to solve the problem, do a switch
			if (update_done || !solveSuccess) {
#pragma omp barrier
				if (systems_to_solve <= 0) break;

				if (!system_changed && systems_to_solve < (init_systems_to_solve >> 1) && solveSuccess) {
				//// if (false) {
#pragma omp barrier
#pragma omp single
					{
						system_changed = true;
						working_thread = 0;
						update_done    = false;
						solveSuccess   = true;
						matrix_updated = false;
					}
					delete mySolver[tid];
					delete mySpmv[tid];
					cusp::io::read_matrix_market_file(A[tid], "/home/ali/CUDA_project/reordering/matrices/88950-lhs.mtx");
					mySolver[tid] = new SaPSolver(numPart, opts);
					mySpmv[tid]   = new CustomSpmv(A[tid]);
					b[tid].resize(A[tid].num_rows);
					cusp::blas::fill(b[tid], 1.0);
					x[tid].resize(A[tid].num_rows);
					
					if (tid == 0) {
						mySolver[0] -> setup(A[0]);
					}
				} else {
#pragma omp single
					{
						update_done = false;
						solveSuccess = true;

						// Copy all results that the other worker are needed for next step
						thrust::copy(b[loc_working_thread].begin(), b[loc_working_thread].end(), b[1-loc_working_thread].begin());

						// Get the up-to-date matrix
						thrust::copy(A[loc_working_thread].values.begin(), A[loc_working_thread].values.end(), A[1-loc_working_thread].values.begin());
					}
				}
			}
		}
	}
	loc_timer.Stop();
	elapsed = loc_timer.getElapsed();
	cout << "Time elapsed: " << elapsed << endl;

	cudaSetDevice(0);
	delete mySolver[0];
	delete mySpmv[0];

	cudaSetDevice(1);
	delete mySolver[1];
	delete mySpmv[1];
	omp_destroy_lock(&g_lock);

	return 0;
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
				int&            systemCount,
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
			case OPT_SYSTEM_NUM:
				systemCount = atoi(args.OptionArg());
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

	if (systemCount <= 0)
		systemCount = 20;

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
	cout << " -n=MATFILE_NEW" << endl;
	cout << " --matrix-file-new=MATFILE_NEW" << endl;
	cout << "        Read the new matrix from the MatrixMarket file MATFILE_NEW." << endl;
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
