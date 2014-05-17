#include <algorithm>
#include <fstream>
#include <cmath>
#include <map>
#include <stdio.h>
#include <stdlib.h>

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
// -----------------------------------------------------------------------------
#ifdef WIN32
#   define ISNAN(A)  (_isnan(A))
#else
#   define ISNAN(A)  (isnan(A))
#endif


// -----------------------------------------------------------------------------
// Typedefs
// -----------------------------------------------------------------------------
typedef double REAL;
typedef double PREC_REAL;

typedef typename cusp::csr_matrix<int, REAL, cusp::device_memory> Matrix;
typedef typename cusp::array1d<REAL, cusp::device_memory>         Vector;
typedef typename cusp::array1d<REAL, cusp::host_memory>           VectorH;

typedef typename spike::Solver<Vector, PREC_REAL>                 SpikeSolver;
typedef typename spike::SpmvCusp<Matrix>                          SpmvFunctor;



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
enum {OPT_HELP, OPT_PART,
      OPT_SPD, OPT_SAVE_MEM,
      OPT_NO_REORDERING, OPT_NO_MC64, OPT_NO_SCALING, OPT_MC64_FIRST_STAGE_ONLY,
      OPT_RTOL, OPT_ATOL, OPT_MAXIT,
      OPT_DROPOFF_FRAC, OPT_MAX_BANDWIDTH,
      OPT_MATFILE, OPT_RHSFILE, 
      OPT_OUTFILE, OPT_FACTORIZATION, OPT_PRECOND,
      OPT_KRYLOV, OPT_SAFE_FACT,
      OPT_CONST_BAND};

// Color to print
enum TestColor {COLOR_NO = 0,
                COLOR_RED,
                COLOR_GREEN} ;

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
	{ OPT_SPD,           "--spd",                SO_NONE    },
	{ OPT_SAVE_MEM,      "--save-mem",           SO_NONE    },
	{ OPT_NO_REORDERING, "--no-reordering",      SO_NONE    },
	{ OPT_NO_MC64,       "--no-mc64",            SO_NONE    },
	{ OPT_NO_SCALING,    "--no-scaling",         SO_NONE    },
	{ OPT_MC64_FIRST_STAGE_ONLY,
	                     "--mc64-first-stage-only", SO_NONE },
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
	SO_END_OF_OPTIONS
};


// -----------------------------------------------------------------------------
// Forward declarations.
// -----------------------------------------------------------------------------
void ShowUsage();
void spikeSetDevice();
bool GetProblemSpecs(int             argc, 
                     char**          argv,
                     string&         fileMat,
                     string&         fileRhs,
                     string&         fileSol,
                     int&            numPart,
                     spike::Options& opts);
void GetRhsVector(const Matrix& A, Vector& b, Vector& x_target);
void PrintStats(bool               success,
                const SpikeSolver& mySolver,
                const SpmvFunctor& mySpmv);
void updateFastest(std::map<std::string, double>& fastest_map, string &mat_name, double time_cur_run, bool solveSuccess = false);

class OutputItem
{
public:
	OutputItem(std::ostream &o): m_o(o), m_additional_item_count(5) {}

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
	string         fileMat;
	string         fileRhs;
	string         fileSol;
	int            numPart;
	spike::Options opts;

	if (!GetProblemSpecs(argc, argv, fileMat, fileRhs, fileSol, numPart, opts))
		return 1;

	opts.testMC64 = true;
	opts.performMC64 = true;

	// Get the device with most available memory.
	spikeSetDevice();

	// Get matrix and rhs.
	Matrix A;
	Vector b;
	Vector x_target;
	Vector delta_x_target;

	cusp::io::read_matrix_market_file(A, fileMat);

	if (fileRhs.length() > 0)
		cusp::io::read_matrix_market_file(b, fileRhs);
	else
		GetRhsVector(A, b, x_target);

	// Create the SPIKE Solver object and the SPMV functor. Perform the solver
	// setup, then solve the linear system using a 0 initial guess.
	// Set the initial guess to the zero vector.
	SpikeSolver  mySolver(numPart, opts);
	SpmvFunctor  mySpmv(A);
	Vector       x(A.num_rows, 0);

	OutputItem outputItem(cout);

	cout << "<tr valign=top>" << endl;

	// Name of matrix
	{
		int i;
		for (i = fileMat.size()-1; i>=0 && fileMat[i] != '/' && fileMat[i] != '\\'; i--);
		i++;
		fileMat = fileMat.substr(i);

		size_t j = fileMat.rfind(".mtx");
		if (j != std::string::npos)
			outputItem( fileMat.substr(0, j));
		else
			outputItem( fileMat);
	}

	// Dimension
	outputItem( A.num_rows);
	// No. of non-zeros
	outputItem( A.num_entries);

	try {
		mySolver.setup(A);
	} catch (const spike::system_error& se) {
		// Make up for the other columns
		for (int i=0; i < outputItem.m_additional_item_count; i++)
			outputItem("");


		cout << "</tr>" << endl;

		return 1;
	}

	{
		spike::Stats stats = mySolver.getStats();

		// Time for MC64 reordering (pre-processing)
		outputItem( stats.time_MC64_pre);
		// Time for MC64 reordering (first stage)
		outputItem( stats.time_MC64_first);
		// Time for MC64 reordering (second stage)
		outputItem( stats.time_MC64_second);
		// Time for MC64 reordering (post-processing)
		outputItem( stats.time_MC64_post);
		// Time for MC64 reordering
		outputItem( stats.time_MC64);

		cout << "</tr>" << endl;
	}

	return 0;
}

void updateFastest(std::map<std::string, double>& fastest_map, string &mat_name, double time_cur_run, bool solveSuccess) {
	std::ofstream fout("../../test_double/fastest_list_tmp.txt", std::ios::out | std::ios::app);
	if (!fout.is_open())
		return;

	if (!solveSuccess) {
		if (fastest_map.find(mat_name) != fastest_map.end())
			fout << mat_name << "\n" << fastest_map[mat_name] << std::endl;
	} else {
		if (fastest_map.find(mat_name) == fastest_map.end() || fastest_map[mat_name] > time_cur_run)
			fastest_map[mat_name] = time_cur_run;

		fout << mat_name << "\n" << fastest_map[mat_name] << std::endl;
	}

		
	fout.close();
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
                string&         fileMat,
                string&         fileRhs,
                string&         fileSol,
                int&            numPart,
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
				break;
			case OPT_SPD:
				opts.isSPD = true;
				break;
			case OPT_SAVE_MEM:
				opts.saveMem = true;
				break;
			case OPT_NO_REORDERING:
				opts.performReorder = false;
				break;
			case OPT_MC64_FIRST_STAGE_ONLY:
				opts.mc64FirstStageOnly = true;
				opts.applyScaling = false;
				break;
			case OPT_NO_MC64:
				opts.performMC64 = false;
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
						opts.factMethod = spike::LU_UL;
					else if (fact == "1" || fact == "LU_LU")
						opts.factMethod = spike::LU_only;
					else
						return false;
				}
				break;
			case OPT_PRECOND:
				{
					string precond = args.OptionArg();
					std::transform(precond.begin(), precond.end(), precond.begin(), ::toupper);
					if (precond == "0" || precond == "SPIKE")
						opts.precondType = spike::Spike;
					else if(precond == "1" || precond == "BLOCK")
						opts.precondType = spike::Block;
					else if(precond == "2" || precond == "NONE")
						opts.precondType = spike::None;
					else
						return false;
				}
				break;
			case OPT_KRYLOV:
				{
					string kry = args.OptionArg();
					std::transform(kry.begin(), kry.end(), kry.begin(), ::toupper);
					if (kry == "0" || kry == "BICGSTAB1")
						opts.solverType = spike::BiCGStab1;
					else if (kry == "1" || kry == "BICGSTAB2")
						opts.solverType = spike::BiCGStab2;
					else if (kry == "2" || kry == "BICGSTAB")
						opts.solverType = spike::BiCGStab;
					else if (kry == "3" || kry == "MINRES")
						opts.solverType = spike::MINRES;
					else if (kry == "4" || kry == "CG")
						opts.solverType = spike::CG;
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

	// For symmetric positive definitive matrix, do not perform MC64
	if (opts.isSPD) {
		opts.performMC64 = false;
		opts.applyScaling = false;
		opts.solverType = spike::CG;
		opts.saveMem = true;
	} else
		opts.saveMem = false;

	// If no reordering, force using constant bandwidth.
	if (!opts.performReorder) {
		opts.variableBandwidth = false;
		opts.performMC64 = false;
	}

	if (!opts.performMC64)
		opts.applyScaling = false;

	// If using variable bandwidth, force using LU factorization.
	if (opts.variableBandwidth)
		opts.factMethod = spike::LU_only;

	return true;
}


// -----------------------------------------------------------------------------
// ShowUsage()
//
// This function displays the correct usage of this program
// -----------------------------------------------------------------------------
void ShowUsage()
{
	cout << "Usage:  driver_test [OPTIONS]" << endl;
	cout << endl;
	cout << " -p=NUM_PARTITIONS" << endl;
	cout << " --num-partitions=NUM_PARTITIONS" << endl;
	cout << "        Specify the number of partitions (default 1)." << endl;
	cout << " --no-reordering" << endl;
	cout << "        Do not perform reordering." << endl;
	cout << " --no-mc64" << endl;
	cout << "        Do not perform MC64 reordering." << endl;
	cout << " --no-scaling" << endl;
	cout << "        Do not perform scaling (ignored if --no-reordering is specified)" << endl;
	cout << " -t=TOLERANCE" << endl;
	cout << " --tolerance=TOLERANCE" << endl;
	cout << " --relTol=TOLERANCE" << endl;
	cout << "        Use relative tolerance TOLERANCE for Krylov stopping criteria (default 1e-6)." << endl;
	cout << " --absTol=TOLERANCE" << endl;
	cout << "        Use absolute tolerance TOLERANCE for Krylov stopping criteria (default 0)." << endl;
	cout << " -i=ITERATIONS" << endl;
	cout << " --max-num-iterations=ITERATIONS" << endl;
	cout << "        Use at most ITERATIONS for KRylov solver (default 100)." << endl;
	cout << " -d=FRACTION" << endl;
	cout << " --drop-off-fraction=FRACTION" << endl;
	cout << "        Drop off-diagonal elements such that FRACTION of the matrix" << endl;
	cout << "        Frobenius norm is ignored (default 0.0 -- i.e. no drop-off)." << endl;
	cout << " -b=MAX_BANDWIDTH" << endl;
	cout << " --max-bandwidth=MAX_BANDWIDTH" << endl;
	cout << "        Drop off elements such that the bandwidth is at most MAX_BANDWIDTH" << endl;
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
	cout << " -k=METHOD" << endl;
	cout << " --krylov-method=METHOD" << endl;
	cout << "        Specify the iterative Krylov solver:" << endl;
	cout << "        METHOD=0 or METHOD=BICGSTAB      use BiCGStab (Cusp)" << endl;
	cout << "        METHOD=1 or METHOD=GMRES         use GMRES (Cusp)" << endl;
	cout << "        METHOD=2 or METHOD=CG            use CG (Cusp)" << endl;
	cout << "        METHOD=3 or METHOD=CR            use CR (Cusp)" << endl;
	cout << "        METHOD=4 or METHOD=BICGSTAB1     use BiCGStab(1) (Spike::GPU)" << endl;
	cout << "        METHOD=5 or METHOD=BICGSTAB2     use BiCGStab(2) (Spike::GPU). This is the default." << endl;
	cout << " --safe-fact" << endl;
	cout << "        Use safe LU-UL factorization." << endl; 
	cout << " --const-band" << endl;
	cout << "        Force using the constant-bandwidth method." << endl; 
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

