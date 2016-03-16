/** \file solver.h
 *  \brief Definition of the main SaP solver class.
 */

#ifndef SAP_SOLVER_H
#define SAP_SOLVER_H

#include <limits>
#include <vector>
#include <string>

#include <sap/common.h>
#include <sap/monitor.h>
#include <sap/precond.h>
#include <sap/bicgstab2.h>
#include <sap/bicgstab.h>
#include <sap/minres.h>
#include <sap/timer.h>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif
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


/** \namespace sap
 * \brief sap is the top-level namespace which contains all SaP functions and types.
 */


namespace sap {

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

    bool                testDB;               /**< Indicate that we are running the test for DB*/
    bool                isSPD;                /**< Indicate whether the matrix is symmetric positive definitive; default: false*/
    bool                saveMem;                /**< (For SPD matrix only) Indicate whether to use memory-saving yet slower mode or not; default: false*/
    bool                performReorder;       /**< Perform matrix reorderings? default: true */
    bool                performDB;            /**< Perform DB reordering? default: true */
    bool                dbFirstStageOnly;     /**< In DB, only the first stage is to be performed? default: false*/
    bool                applyScaling;         /**< Apply DB scaling? default: true */
    int                 maxBandwidth;         /**< Maximum half-bandwidth; default: INT_MAX */
    int                 gpuCount;             /**< Number of GPU expected to use; default: 1 */
    double              dropOffFraction;      /**< Maximum fraction of the element-wise matrix 1-norm that can be dropped-off; default: 0 */

    FactorizationMethod factMethod;           /**< Diagonal block factorization method; default: LU_only */
    PreconditionerType  precondType;          /**< Preconditioner type; default: Spike */
    bool                safeFactorization;    /**< Use safe factorization (diagonal boosting)? default: false */
    bool                variableBandwidth;    /**< Allow variable partition bandwidths? default: true */
    bool                trackReordering;      /**< Keep track of the reordering information? default: false */

    bool                useBCR;

    int                 ilu_level;            /**< Indicate the level of ILU, a minus value means complete LU is applied; default: -1*/
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

    double      time_DB;                /**< Time to do DB reordering. */
    double      time_DB_pre;            /**< Time to do DB reordering (pre-processing). */
    double      time_DB_first;          /**< Time to do DB reordering (first stage). */
    double      time_DB_second;         /**< Time to do DB reordering (second stage). */
    double      time_DB_post;           /**< Time to do DB reordering (post-processing). */
    double      time_reorder;           /**< Time to do DB reordering. */
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

    double      time_bcr_lu;
    double      time_bcr_sweep_deflation;
    double      time_bcr_mat_mul_deflation;
    double      time_bcr_sweep_inflation;
    double      time_bcr_mv_inflation;

    int         bandwidthReorder;       /**< Half-bandwidth after reordering. */
    int         bandwidthDB;            /**< Half-bandwidth after DB. */
    int         bandwidth;              /**< Half-bandwidth after reordering and drop-off. */
    double      nuKf;                   /**< Non-uniform K factor. Indicates whether the K changes a lot from row to row. */
    double      flops_LU;               /**< FLOPs of LU*/

    int         numPartitions;          /**< Actual number of partitions used in the SaP factorization */
    double      actualDropOff;          /**< Actual fraction of the element-wise matrix 1-norm dropped off. */

    float       numIterations;          /**< Number of iterations required for iterative solver to converge. */
    double      rhsNorm;                /**< RHS norm (i.e. ||b||_2). */
    double      residualNorm;           /**< Final residual norm (i.e. ||b-Ax||_2). */
    double      relResidualNorm;        /**< Final relative residual norm (i.e. ||b-Ax||_2 / ||b||_2)*/

    int         actual_nnz;
};


/// Main SaP::GPU solver.
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
    ~Solver() {
        if (m_p_monitor != NULL) {
            delete m_p_monitor;
            m_p_monitor = NULL;
        } 
        if (m_p_bicgstabl_monitor != NULL){
            delete m_p_bicgstabl_monitor;
            m_p_bicgstabl_monitor = NULL;
        }
    }

    template <typename Matrix>
    bool setup(const Matrix& A);

    template <typename Array1>
    bool update(const Array1& entries);

    template <typename SpmvOperator>
    bool solve(SpmvOperator&  spmv,
               const Array&   b,
               Array&         x);

    /// Extract solver statistics.
    const Stats&       getStats() const          {return m_stats;}
    int                getMonitorCode() const    {
        if (m_p_monitor != NULL) {
            return m_p_monitor -> getCode();
        }
        return m_p_bicgstabl_monitor->getCode();
    }
    const std::string& getMonitorMessage() const {
        if (m_p_monitor != NULL) {
            return m_p_monitor -> getMessage();
        }
        return m_p_bicgstabl_monitor->getMessage();
    }

private:
    typedef typename Array::value_type    SolverValueType;
    typedef typename Array::memory_space  MemorySpace;

    typedef typename cusp::array1d<SolverValueType, MemorySpace>        SolverVector;
    typedef typename cusp::array1d<PrecValueType,   MemorySpace>        PrecVector;

    typedef typename cusp::array1d<PrecValueType,   cusp::host_memory>  PrecVectorH;
    typedef typename cusp::array1d<int,             cusp::host_memory>  IntVectorH;

    typedef typename cusp::coo_matrix<int, PrecValueType, cusp::host_memory>  PrecMatrixCooH;


    KrylovSolverType                    m_solver;
    Monitor<SolverVector>*              m_p_monitor;
    BiCGStabLMonitor<SolverVector>*     m_p_bicgstabl_monitor;
    Precond<PrecVector>                 m_precond;

    int                                 m_n;
    int                                 m_nnz;
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
:   solverType(BiCGStab2),
    maxNumIterations(100),
    gpuCount(1),
    relTol(1e-6),
    absTol(0),
    testDB(false),
    isSPD(false),
    saveMem(false),
    performReorder(true),
    performDB(true),
    dbFirstStageOnly(false),
    applyScaling(true),
    maxBandwidth(std::numeric_limits<int>::max()),
    dropOffFraction(0),
    factMethod(LU_only),
    precondType(Spike),
    safeFactorization(false),
    variableBandwidth(true),
    trackReordering(false),
    useBCR(false),
    ilu_level(-1)
{
}

/**
 * This is the constructor for the Stats class. It initializes all
 * timing and performance measures.
 */
inline
Stats::Stats()
:   timeSetup(0),
    timeSolve(0),
    time_DB(0),
    time_DB_pre(0),
    time_DB_first(0),
    time_DB_second(0),
    time_DB_post(0),
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
    bandwidthDB(0),
    bandwidth(0),
    nuKf(0),
    flops_LU(0),
    numPartitions(0),
    actualDropOff(0),
    numIterations(0),
    rhsNorm(std::numeric_limits<double>::max()),
    residualNorm(std::numeric_limits<double>::max()),
    relResidualNorm(std::numeric_limits<double>::max())
{
}


/// SaP solver constructor.
/**
 * This is the constructor for the Solver class. It specifies the requested number
 * of partitions and the set of solver options.
 */
template <typename Array, typename PrecValueType>
Solver<Array, PrecValueType>::Solver(int             numPartitions,
                                     const Options&  opts)
:   m_precond(numPartitions, opts.isSPD, opts.saveMem, opts.performReorder, opts.testDB, opts.performDB, opts.dbFirstStageOnly, opts.applyScaling,
              opts.dropOffFraction, opts.maxBandwidth, opts.gpuCount, opts.factMethod, opts.precondType, 
              opts.safeFactorization, opts.variableBandwidth, opts.trackReordering, opts.useBCR, opts.ilu_level, opts.relTol),
    m_solver(opts.solverType),
    m_trackReordering(opts.trackReordering),
    m_setupDone(false)
{
    if (m_solver == BiCGStab1 || m_solver == BiCGStab2) {
        m_p_monitor = NULL;
        m_p_bicgstabl_monitor = new BiCGStabLMonitor<SolverVector>(
            opts.maxNumIterations,
            8,
            opts.relTol,
            opts.absTol
        );
    } else {
        m_p_bicgstabl_monitor = NULL;
        m_p_monitor = new Monitor<SolverVector>(
            opts.maxNumIterations,
            opts.relTol,
            opts.absTol
        );
    }
}


/// Preconditioner setup.
/**
 * This function performs the initial setup for the SaP solver. It prepares
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
    m_n   = A.num_rows;
    m_nnz = A.num_entries;

    CPUTimer timer;

    timer.Start();

    m_precond.setup(A);

    timer.Stop();

    m_stats.timeSetup = timer.getElapsed();

    m_stats.bandwidthReorder = m_precond.getBandwidthReordering();
    m_stats.bandwidth = m_precond.getBandwidth();
    m_stats.bandwidthDB= m_precond.getBandwidthDB();
    m_stats.nuKf = (double) cusp::blas::nrm1(m_precond.m_ks_row_host) + cusp::blas::nrm1(m_precond.m_ks_col_host);
    m_stats.flops_LU = 0;
    {
        int n = m_precond.m_ks_row_host.size();
        for (int i=0; i<n; i++)
            m_stats.flops_LU += (double)(m_precond.m_ks_row_host[i]) * (m_precond.m_ks_col_host[i]);
    }
    m_stats.numPartitions = m_precond.getNumPartitions();
    m_stats.actualDropOff = m_precond.getActualDropOff();
    m_stats.time_DB = m_precond.getTimeDB();
    m_stats.time_DB_pre = m_precond.getTimeDBPre();
    m_stats.time_DB_first = m_precond.getTimeDBFirst();
    m_stats.time_DB_second = m_precond.getTimeDBSecond();
    m_stats.time_DB_post = m_precond.getTimeDBPost();
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

    m_stats.actual_nnz  = m_precond.getActualNumNonZeros();

    if (m_stats.bandwidth == 0)
        m_stats.nuKf = 0.0;
    else
        m_stats.nuKf = (2.0 * m_stats.bandwidth * m_n- m_stats.nuKf) / (2.0 * m_stats.bandwidth * m_n);

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

    // If the matrix pattern has actually changed, FIXME: do we need more checking?
    if (entries.size() != m_nnz)
        return false;

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
 *         to calculate sparse matrix-vector product. See sap::SpmvCusp
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
    if (m_p_monitor != NULL) {
        m_p_monitor -> init(b_vector);
    } else {
        m_p_bicgstabl_monitor -> init(b_vector);
    }

    CPUTimer timer;

    timer.Start();

    switch(m_solver)
    {
        // CUSP Krylov solvers
        case BiCGStab_C:
            cusp::krylov::bicgstab(spmv, x_vector, b_vector, *m_p_monitor, m_precond);
            break;
        case GMRES_C:
            cusp::krylov::gmres(spmv, x_vector, b_vector, 50, *m_p_monitor, m_precond);
            break;
        case CG_C:
            cusp::krylov::cg(spmv, x_vector, b_vector, *m_p_monitor, m_precond);
            break;
        case CR_C:
            cusp::krylov::cr(spmv, x_vector, b_vector, *m_p_monitor, m_precond);
            break;

        // SaP Krylov solvers
        case BiCGStab1:
            sap::bicgstab1(spmv, x_vector, b_vector, *m_p_bicgstabl_monitor, m_precond);
            break;
        case BiCGStab2:
            sap::bicgstab2(spmv, x_vector, b_vector, *m_p_bicgstabl_monitor, m_precond);
            break;
        case BiCGStab:
            sap::bicgstab(spmv, x_vector, b_vector, *m_p_monitor, m_precond);
            break;
        case MINRES:
            sap::minres(spmv, x_vector, b_vector, *m_p_monitor, m_precond);
            break;
    }

    thrust::copy(x_vector.begin(), x_vector.end(), x.begin());
    timer.Stop();

    m_stats.timeSolve = timer.getElapsed();
    if (m_p_monitor != NULL) {
        m_stats.rhsNorm = m_p_monitor -> getRHSNorm();
        m_stats.residualNorm = m_p_monitor -> getResidualNorm();
        m_stats.relResidualNorm = m_p_monitor -> getRelResidualNorm();
        m_stats.numIterations = m_p_monitor -> getNumIterations();
    } else {
        m_stats.rhsNorm = m_p_bicgstabl_monitor -> getRHSNorm();
        m_stats.residualNorm = m_p_bicgstabl_monitor -> getResidualNorm();
        m_stats.relResidualNorm = m_p_bicgstabl_monitor -> getRelResidualNorm();
        m_stats.numIterations = m_p_bicgstabl_monitor -> getNumIterations();
    }

    m_stats.time_shuffle = m_precond.getTimeShuffle();

    m_stats.time_bcr_lu = m_precond.getTimeBCRLU();
    m_stats.time_bcr_sweep_deflation = m_precond.getTimeBCRSweepDeflation();
    m_stats.time_bcr_mat_mul_deflation = m_precond.getTimeBCRMatMulDeflation();
    m_stats.time_bcr_sweep_inflation = m_precond.getTimeBCRSweepInflation();
    m_stats.time_bcr_mv_inflation = m_precond.getTimeBCRMVInflation();

    if (m_p_monitor != NULL) {
        return m_p_monitor -> converged();
    }
    return m_p_bicgstabl_monitor -> converged();
}


} // namespace sap


#endif
