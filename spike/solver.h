/** \file solver.h
 *  \brief Definition of the main Spike solver class.
 */

#ifndef SPIKE_SOLVER_H
#define SPIKE_SOLVER_H

#include <limits>
#include <vector>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/blas.h>
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/cr.h>
#include <cusp/krylov/gmres.h>

#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/logical.h>


#include <spike/common.h>
#include <spike/monitor.h>
#include <spike/precond.h>
#include <spike/bicgstab2.h>
#include <spike/bicgstab.h>
#include <spike/minres.h>
#include <spike/timer.h>


/** \namespace spike
 * \brief spike is the top-level namespace which contains all Spike functions and types.
 */


namespace spike {

/// Input solver options.
/**
 * This structure encapsulates all solver options and specifies the methods and
 * parameters used in the iterative solver and the preconditioner.
 */
struct Options
{
	Options();

	KrylovSolverType    solverType;           /**< Krylov method to use; default: BiCGStab2 */
	int                 maxNumIterations;     /**< Maximum number of iterations; default: 100 */
	double              relTol;               /**< Relative tolerance; default: 1e-6 */
	double              absTol;               /**< Absolute tolerance; default: 0 */

	bool                testMC64;             /**< Indicate that we are running the test for MC64*/
	bool                isSPD;                /**< Indicate whether the matrix is symmetric positive definitive; default: false*/
	bool                saveMem;                /**< (For SPD matrix only) Indicate whether to use memory-saving yet slower mode or not; default: false*/
	bool                performReorder;       /**< Perform matrix reorderings? default: true */
	bool                performMC64;          /**< Perform MC64 reordering? default: true */
	bool                mc64FirstStageOnly;   /**< In MC64, only the first stage is to be performed? default: false*/
	bool                applyScaling;         /**< Apply MC64 scaling? default: true */
	int                 maxBandwidth;         /**< Maximum half-bandwidth; default: INT_MAX */
	double              dropOffFraction;      /**< Maximum fraction of the element-wise matrix 1-norm that can be dropped-off; default: 0 */

	FactorizationMethod factMethod;           /**< Diagonal block factorization method; default: LU_only */
	PreconditionerType  precondType;          /**< Preconditioner type; default: Spike */
	bool                safeFactorization;    /**< Use safe factorization (diagonal boosting)? default: false */
	bool                variableBandwidth;    /**< Allow variable partition bandwidths? default: true */
	bool                trackReordering;      /**< Keep track of the reordering information? default: false */
};


/// Output solver statistics.
/**
 * This structure encapsulates all solver statistics, both from the iterative
 * solver and the preconditioner.
 */
struct Stats
{
	Stats();

	double      timeSetup;              /**< Time to set up the preconditioner. */
	double      timeUpdate;             /**< Time to update the preconditioner. */
	double      timeSolve;              /**< Time for Krylov solve. */

	double      time_MC64;              /**< Time to do MC64 reordering. */
	double      time_MC64_pre;          /**< Time to do MC64 reordering (pre-processing). */
	double      time_MC64_first;        /**< Time to do MC64 reordering (first stage). */
	double      time_MC64_second;       /**< Time to do MC64 reordering (second stage). */
	double      time_MC64_post;         /**< Time to do MC64 reordering (post-processing). */
	double      time_reorder;           /**< Time to do reordering. */
	double      time_dropOff;           /**< Time for drop-off*/
	double      time_cpu_assemble;      /**< Time on CPU to assemble the banded matrix and off-diagonal spikes. */
	double      time_transfer;          /**< Time to transfer data from CPU to GPU. */
	double      time_toBanded;          /**< Time to form banded matrix when reordering is disabled.*/ /*TODO: combine this with time_cpu_assemble*/
	double      time_offDiags;          /**< Time to compute off-diagonal spike matrices on GPU. */
	double      time_bandLU;            /**< Time for LU factorization of diagonal blocks. */
	double      time_bandUL;            /**< Time for UL factorization of diagonal blocks(in LU_UL method only). */
	double      time_fullLU;            /**< Time for LU factorization of the reduced matrix R. */
	double      time_assembly;          /**< Time for assembling off-diagonal matrices (including solving multiple RHS) */

	double      time_shuffle;           /**< Total time to do vector reordering and scaling. */

	int         bandwidthReorder;       /**< Half-bandwidth after reordering. */
	int         bandwidthMC64;          /**< Half-bandwidth after MC64. */
	int         bandwidth;              /**< Half-bandwidth after reordering and drop-off. */
	double      nuKf;                   /**< Non-uniform K factor. Indicates whether the K changes a lot from row to row. */
	double      flops_LU;               /**< FLOPs of LU*/

	int         numPartitions;          /**< Actual number of partitions used in the Spike factorization */
	double      actualDropOff;          /**< Actual fraction of the element-wise matrix 1-norm dropped off. */

	float       numIterations;          /**< Number of iterations required for iterative solver to converge. */
	double      residualNorm;           /**< Final residual norm (i.e. ||b-Ax||_2). */
	double      relResidualNorm;        /**< Final relative residual norm (i.e. ||b-Ax||_2 / ||b||_2)*/
};


/// Main SPIKE::GPU solver.
/** 
 * This class is the public interface to the Spike-preconditioned
 * Krylov iterative solver.
 *
 * \tparam Array is the array type for the linear system solution.
 *         (both cusp::array1d and cusp::array1d_view are valid).
 * \tparam PrecValueType is the floating point type used in the preconditioner
 *         (to support mixed-precision calculations).
 */
template <typename Array, typename PrecValueType>
class Solver
{
public:
	Solver(int             numPartitions,
	       const Options&  opts);

	template <typename Matrix>
	bool setup(const Matrix& A);

	template <typename Array1>
	bool update(const Array1& entries);

	template <typename SpmvOperator>
	bool solve(SpmvOperator&  spmv,
	           const Array&   b,
	           Array&         x);

	/// Extract solver statistics.
	const Stats&  getStats() const {return m_stats;}


private:
	typedef typename Array::value_type    SolverValueType;
	typedef typename Array::memory_space  MemorySpace;

	typedef typename cusp::array1d<SolverValueType, MemorySpace>        SolverVector;
	typedef typename cusp::array1d<PrecValueType,   MemorySpace>        PrecVector;

	typedef typename cusp::array1d<PrecValueType,   cusp::host_memory>  PrecVectorH;
	typedef typename cusp::array1d<int,             cusp::host_memory>  IntVectorH;

	typedef typename cusp::coo_matrix<int, PrecValueType, cusp::host_memory>  PrecMatrixCooH;


	KrylovSolverType                    m_solver;
	Monitor<SolverVector>               m_monitor;
	Precond<PrecVector>                 m_precond;

	int                                 m_n;
	bool                                m_trackReordering;
	bool                                m_setupDone;

	Stats                               m_stats;

public:
	// FIXME: this should only be used in nightly test, remove this
	const Precond<PrecVector>&          getPreconditioner() const
										{return m_precond;}
};


/**
 * This is the constructor for the Options class. It sets default values for
 * all options.
 */
inline
Options::Options()
:	solverType(BiCGStab2),
	maxNumIterations(100),
	relTol(1e-6),
	absTol(0),
	testMC64(false),
	isSPD(false),
	saveMem(false),
	performReorder(true),
	performMC64(true),
	mc64FirstStageOnly(false),
	applyScaling(true),
	maxBandwidth(std::numeric_limits<int>::max()),
	dropOffFraction(0),
	factMethod(LU_only),
	precondType(Spike),
	safeFactorization(false),
	variableBandwidth(true),
	trackReordering(false)
{
}

/**
 * This is the constructor for the Stats class. It initializes all
 * timing and performance measures.
 */
inline
Stats::Stats()
:	timeSetup(0),
	timeSolve(0),
	time_MC64(0),
	time_MC64_pre(0),
	time_MC64_first(0),
	time_MC64_second(0),
	time_MC64_post(0),
	time_reorder(0),
	time_dropOff(0),
	time_cpu_assemble(0),
	time_transfer(0),
	time_toBanded(0),
	time_offDiags(0),
	time_bandLU(0),
	time_bandUL(0),
	time_assembly(0),
	time_fullLU(0),
	time_shuffle(0),
	bandwidthReorder(0),
	bandwidthMC64(0),
	bandwidth(0),
	nuKf(0),
	flops_LU(0),
	numPartitions(0),
	actualDropOff(0),
	numIterations(0),
	residualNorm(std::numeric_limits<double>::max()),
	relResidualNorm(std::numeric_limits<double>::max())
{
}


/// Spike solver constructor.
/**
 * This is the constructor for the Solver class. It specifies the requested number
 * of partitions and the set of solver options.
 */
template <typename Array, typename PrecValueType>
Solver<Array, PrecValueType>::Solver(int             numPartitions,
                                     const Options&  opts)
:	m_monitor(opts.maxNumIterations, opts.relTol, opts.absTol),
	m_precond(numPartitions, opts.isSPD, opts.saveMem, opts.performReorder, opts.testMC64, opts.performMC64, opts.mc64FirstStageOnly, opts.applyScaling,
	          opts.dropOffFraction, opts.maxBandwidth, opts.factMethod, opts.precondType, 
	          opts.safeFactorization, opts.variableBandwidth, opts.trackReordering),
	m_solver(opts.solverType),
	m_trackReordering(opts.trackReordering),
	m_setupDone(false)
{
}


/// Preconditioner setup.
/**
 * This function performs the initial setup for the Spike solver. It prepares
 * the preconditioner based on the specified matrix A (which may be the system
 * matrix, or some approximation to it).
 *
 * \tparam Matrix is the sparse matrix type used in the preconditioner.
 */
template <typename Array, typename PrecValueType>
template <typename Matrix>
bool
Solver<Array, PrecValueType>::setup(const Matrix& A)
{
	m_n = A.num_rows;

	CPUTimer timer;

	timer.Start();

	m_precond.setup(A);

	timer.Stop();

	m_stats.timeSetup = timer.getElapsed();

	m_stats.bandwidthReorder = m_precond.getBandwidthReordering();
	m_stats.bandwidth = m_precond.getBandwidth();
	m_stats.bandwidthMC64 = m_precond.getBandwidthMC64();
	m_stats.nuKf = cusp::blas::nrm1(m_precond.m_ks_row_host) + cusp::blas::nrm1(m_precond.m_ks_col_host);
	m_stats.flops_LU = 0;
	{
		int n = m_precond.m_ks_row_host.size();
		for (int i=0; i<n; i++)
			m_stats.flops_LU += (double)(m_precond.m_ks_row_host[i]) * (m_precond.m_ks_col_host[i]);
	}
	m_stats.numPartitions = m_precond.getNumPartitions();
	m_stats.actualDropOff = m_precond.getActualDropOff();
	m_stats.time_MC64 = m_precond.getTimeMC64();
	m_stats.time_MC64_pre = m_precond.getTimeMC64Pre();
	m_stats.time_MC64_first = m_precond.getTimeMC64First();
	m_stats.time_MC64_second = m_precond.getTimeMC64Second();
	m_stats.time_MC64_post = m_precond.getTimeMC64Post();
	m_stats.time_reorder = m_precond.getTimeReorder();
	m_stats.time_dropOff = m_precond.getTimeDropOff();
	m_stats.time_cpu_assemble = m_precond.getTimeCPUAssemble();
	m_stats.time_transfer = m_precond.getTimeTransfer();
	m_stats.time_toBanded = m_precond.getTimeToBanded();
	m_stats.time_offDiags = m_precond.getTimeCopyOffDiags();
	m_stats.time_bandLU = m_precond.getTimeBandLU();
	m_stats.time_bandUL = m_precond.getTimeBandUL();
	m_stats.time_assembly = m_precond.gettimeAssembly();
	m_stats.time_fullLU = m_precond.getTimeFullLU();

	if (m_stats.bandwidth == 0)
		m_stats.nuKf = 0.0;
	else
		m_stats.nuKf = (2 * m_stats.bandwidth * m_n- m_stats.nuKf) / (2 * m_stats.bandwidth * m_n);

	if (m_stats.time_bandLU == 0)
		m_stats.flops_LU = 0;
	else
		m_stats.flops_LU /= m_stats.time_bandLU * 1e6;

	m_setupDone = true;

	return true;
}


/// Preconditioner update.
/**
 * This function updates the Spike preconditioner assuming that the reordering
 * information generated when the preconditioner was initially set up is still
 * valid.  The diagonal blocks and off-diagonal spike blocks are updates based
 * on the provided matrix non-zero entries.
 * 
 * An exception is thrown if this call was not preceeded by a call to
 * Solver::setup() or if reordering tracking was not enabled through the solver
 * options.
 *
 * \tparam Array1 is the vector type for the non-zero entries of the updated
 *         matrix (both cusp::array1d and cusp::array1d_view are allowed).
 */
template <typename Array, typename PrecValueType>
template <typename Array1>
bool
Solver<Array, PrecValueType>::update(const Array1& entries)
{
	// Check if this call to update() is legal.
	if (!m_setupDone)
		throw system_error(system_error::Illegal_update, "Illegal call to update() before setup().");

	if (!m_trackReordering)
		throw system_error(system_error::Illegal_update, "Illegal call to update() with reordering tracking disabled.");


	// Update the preconditioner.
	CPUTimer timer;
	timer.Start();

	{
		PrecVector tmp_entries = entries;
		
		m_precond.update(tmp_entries);
	}

	timer.Stop();

	m_stats.timeUpdate = timer.getElapsed();

	m_stats.time_reorder = 0;
	m_stats.time_cpu_assemble = m_precond.getTimeCPUAssemble();
	m_stats.time_transfer = m_precond.getTimeTransfer();
	m_stats.time_toBanded = m_precond.getTimeToBanded();
	m_stats.time_offDiags = m_precond.getTimeCopyOffDiags();
	m_stats.time_bandLU = m_precond.getTimeBandLU();
	m_stats.time_bandUL = m_precond.getTimeBandUL();
	m_stats.time_assembly = m_precond.gettimeAssembly();
	m_stats.time_fullLU = m_precond.getTimeFullLU();

	return true;
}


/// Linear system solve
/**
 * This function solves the system Ax=b, for given matrix A and right-handside
 * vector b.
 *
 * An exception is throw if this call was not preceeded by a call to
 * Solver::setup().
 *
 * \tparam SpmvOperator is a functor class which implements the operator()
 *         to calculate sparse matrix-vector product. See spike::SpmvCusp
 *         for an example.
 */
template <typename Array, typename PrecValueType>
template <typename SpmvOperator>
bool
Solver<Array, PrecValueType>::solve(SpmvOperator&       spmv,
                                    const Array&        b,
                                    Array&              x)
{
	// Check if this call to solve() is legal.
	if (!m_setupDone)
		throw system_error(system_error::Illegal_solve, "Illegal call to solve() before setup().");

	SolverVector b_vector = b;
	SolverVector x_vector = x;


	// Solve the linear system.
	m_monitor.init(b_vector);

	CPUTimer timer;

	timer.Start();

	switch(m_solver)
	{
		// CUSP Krylov solvers
		case BiCGStab_C:
			cusp::krylov::bicgstab(spmv, x_vector, b_vector, m_monitor, m_precond);
			break;
		case GMRES_C:
			cusp::krylov::gmres(spmv, x_vector, b_vector, 50, m_monitor, m_precond);
			break;
		case CG_C:
			cusp::krylov::cg(spmv, x_vector, b_vector, m_monitor, m_precond);
			break;
		case CR_C:
			cusp::krylov::cr(spmv, x_vector, b_vector, m_monitor, m_precond);
			break;

		// SPIKE Krylov solvers
		case BiCGStab1:
			spike::bicgstab1(spmv, x_vector, b_vector, m_monitor, m_precond);
			break;
		case BiCGStab2:
			spike::bicgstab2(spmv, x_vector, b_vector, m_monitor, m_precond);
			break;
		case BiCGStab:
			spike::bicgstab(spmv, x_vector, b_vector, m_monitor, m_precond);
			break;
		case MINRES:
			spike::minres(spmv, x_vector, b_vector, m_monitor, m_precond);
			break;
	}

	thrust::copy(x_vector.begin(), x_vector.end(), x.begin());
	timer.Stop();

	m_stats.timeSolve = timer.getElapsed();
	m_stats.residualNorm = m_monitor.getResidualNorm();
	m_stats.relResidualNorm = m_monitor.getRelResidualNorm();
	m_stats.numIterations = m_monitor.getNumIterations();

	m_stats.time_shuffle = m_precond.getTimeShuffle();

	return m_monitor.converged();
}


} // namespace spike


#endif
