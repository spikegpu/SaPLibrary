#ifndef SPIKE_SOLVER_H
#define SPIKE_SOLVER_H

#include <limits>
#include <vector>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/blas.h>
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>

#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/logical.h>


#include <spike/common.h>
#include <spike/components.h>
#include <spike/monitor.h>
#include <spike/precond.h>
#include <spike/bicgstab2.h>
#include <spike/timer.h>


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
	double              tolerance;            /**< Relative tolerance; default: 1e-6 */

	bool                performReorder;       /**< Perform matrix reorderings? default: true */
	bool                performMC64;          /**< Perform MC64 reordering? default: true */
	bool                applyScaling;         /**< Apply MC64 scaling? default: true */
	int                 maxBandwidth;         /**< Maximum half-bandwidth; default: INT_MAX */
	double              dropOffFraction;      /**< Maximum fraction of the element-wise matrix 1-norm that can be dropped-off; default: 0 */

	FactorizationMethod factMethod;           /**< Diagonal block factorization method; default: LU_only */
	PreconditionerType  precondType;          /**< Preconditioner type; default: Spike */
	bool                safeFactorization;    /**< Use safe factorization (diagonal boosting)? default: false */
	bool                variableBandwidth;    /**< Allow variable partition bandwidths? default: true */
	bool                singleComponent;      /**< Disable check for disconnected components? default: false */
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

	double      time_reorder;           /**< Time to do reordering. */
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
 */
template <typename Array, typename PrecValueType>
class Solver
{
public:
	Solver(int             numPartitions,
	       const Options&  opts);
	~Solver();

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
	std::vector<Precond<PrecVector>*>   m_precond_pointers;
	std::vector<IntVectorH>             m_comp_reorderings;
	IntVectorH                          m_comp_perms;
	IntVectorH                          m_compIndices;
	IntVectorH                          m_compMap;

	int                                 m_n;
	bool                                m_singleComponent;
	bool                                m_trackReordering;
	bool                                m_setupDone;

	Stats                               m_stats;
};


/**
 * This is the constructor for the Options class. It sets default values for
 * all options.
 */
inline
Options::Options()
:	solverType(BiCGStab2),
	maxNumIterations(100),
	tolerance(1e-6),
	performReorder(true),
	performMC64(true),
	applyScaling(true),
	maxBandwidth(std::numeric_limits<int>::max()),
	dropOffFraction(0),
	factMethod(LU_only),
	precondType(Spike),
	safeFactorization(false),
	variableBandwidth(true),
	singleComponent(false),
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
	time_reorder(0),
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
:	m_monitor(opts.maxNumIterations, opts.tolerance),
	m_precond(numPartitions, opts.performReorder, opts.performMC64, opts.applyScaling,
	          opts.dropOffFraction, opts.maxBandwidth, opts.factMethod, opts.precondType, 
	          opts.safeFactorization, opts.variableBandwidth, opts.trackReordering),
	m_solver(opts.solverType),
	m_singleComponent(opts.singleComponent),
	m_trackReordering(opts.trackReordering),
	m_setupDone(false)
{
}


/// Spike solver destructor.
/**
 * This is the destructor for the Solver class. It frees the preconditioner objects.
 */
template <typename Array, typename PrecValueType>
Solver<Array, PrecValueType>::~Solver()
{
		for (size_t i = 0; i < m_precond_pointers.size(); i++)
			delete m_precond_pointers[i];
		m_precond_pointers.clear();
}


/// Preconditioner setup.
/**
 * This function performs the initial setup for the Spike solver. It prepares
 * the preconditioner based on the specified matrix A (which may be the system
 * matrix, or some approximation to it).
 */
template <typename Array, typename PrecValueType>
template <typename Matrix>
bool
Solver<Array, PrecValueType>::setup(const Matrix& A)
{
	m_n = A.num_rows;
	size_t nnz = A.num_entries;

	CPUTimer timer;

	timer.Start();

	PrecMatrixCooH Acoo;
	
	Components sc(m_n);
	size_t     numComponents;

	if (m_singleComponent)
		numComponents = 1;
	else {
		Acoo = A;
		for (size_t i = 0; i < nnz; i++)
			sc.combineComponents(Acoo.row_indices[i], Acoo.column_indices[i]);
		sc.adjustComponentIndices();
		numComponents = sc.m_numComponents;
	}

	if (m_trackReordering)
		m_compMap.resize(nnz);

	if (numComponents > 1) {
		m_compIndices = sc.m_compIndices;

		std::vector<PrecMatrixCooH> coo_matrices(numComponents);

		m_comp_reorderings.resize(numComponents);
		if (m_comp_perms.size() != m_n)
			m_comp_perms.resize(m_n, -1);
		else
			cusp::blas::fill(m_comp_perms, -1);

		for (size_t i=0; i < numComponents; i++)
			m_comp_reorderings[i].clear();

		IntVectorH cur_indices(numComponents, 0);

		IntVectorH visited(m_n, 0);

		for (size_t i = 0; i < nnz; i++) {
			int from = Acoo.row_indices[i];
			int to = Acoo.column_indices[i];
			int compIndex = sc.m_compIndices[from];

			visited[from] = visited[to] = 1;

			if (m_comp_perms[from] < 0) {
				m_comp_perms[from] = cur_indices[compIndex];
				m_comp_reorderings[compIndex].push_back(from);
				cur_indices[compIndex] ++;
			}

			if (m_comp_perms[to] < 0) {
				m_comp_perms[to] = cur_indices[compIndex];
				m_comp_reorderings[compIndex].push_back(to);
				cur_indices[compIndex] ++;
			}

			PrecMatrixCooH& cur_matrix = coo_matrices[compIndex];

			cur_matrix.row_indices.push_back(m_comp_perms[from]);
			cur_matrix.column_indices.push_back(m_comp_perms[to]);
			cur_matrix.values.push_back(Acoo.values[i]);

			if (m_trackReordering)
				m_compMap[i] = compIndex;
		}

		if (thrust::any_of(visited.begin(), visited.end(), thrust::logical_not<int>() ))
			throw system_error(system_error::Matrix_singular, "Singular matrix found");

		for (size_t i = 0; i < numComponents; i++) {
			PrecMatrixCooH& cur_matrix = coo_matrices[i];

			cur_matrix.num_entries = cur_matrix.values.size();
			cur_matrix.num_rows = cur_matrix.num_cols = cur_indices[i];

			if (m_precond_pointers.size() <= i)
				m_precond_pointers.push_back(new Precond<PrecVector>(m_precond));

			m_precond_pointers[i]->setup(cur_matrix);
		}
	} else {
		m_compIndices.resize(m_n, 0);
		m_comp_reorderings.resize(1);
		m_comp_reorderings[0].resize(m_n);
		m_comp_perms.resize(m_n);

		if (m_trackReordering)
			cusp::blas::fill(m_compMap, 0);

		thrust::sequence(m_comp_perms.begin(), m_comp_perms.end());
		thrust::sequence(m_comp_reorderings[0].begin(), m_comp_reorderings[0].end());

		if (m_precond_pointers.size() == 0)
			m_precond_pointers.push_back(new Precond<PrecVector>(m_precond));

		m_precond_pointers[0]->setup(A);
	}

	timer.Stop();

	m_stats.timeSetup = timer.getElapsed();

	m_stats.bandwidthReorder = m_precond_pointers[0]->getBandwidthReordering();
	m_stats.bandwidth = m_precond_pointers[0]->getBandwidth();
	m_stats.bandwidthMC64 = m_precond_pointers[0]->getBandwidthMC64();
	m_stats.numPartitions = m_precond_pointers[0]->getNumPartitions();
	m_stats.actualDropOff = m_precond_pointers[0]->getActualDropOff();
	m_stats.time_reorder = m_precond_pointers[0]->getTimeReorder();
	m_stats.time_cpu_assemble = m_precond_pointers[0]->getTimeCPUAssemble();
	m_stats.time_transfer = m_precond_pointers[0]->getTimeTransfer();
	m_stats.time_toBanded = m_precond_pointers[0]->getTimeToBanded();
	m_stats.time_offDiags = m_precond_pointers[0]->getTimeCopyOffDiags();
	m_stats.time_bandLU = m_precond_pointers[0]->getTimeBandLU();
	m_stats.time_bandUL = m_precond_pointers[0]->getTimeBandUL();
	m_stats.time_assembly = m_precond_pointers[0]->gettimeAssembly();
	m_stats.time_fullLU = m_precond_pointers[0]->getTimeFullLU();

	for (size_t i=1; i < numComponents; i++) {
		if (m_stats.bandwidthReorder < m_precond_pointers[i]->getBandwidthReordering())
			m_stats.bandwidthReorder = m_precond_pointers[i]->getBandwidthReordering();
		if (m_stats.bandwidth < m_precond_pointers[i]->getBandwidth())
			m_stats.bandwidth = m_precond_pointers[i]->getBandwidth();
		if (m_stats.bandwidthMC64 < m_precond_pointers[i]->getBandwidthMC64())
			m_stats.bandwidthMC64 = m_precond_pointers[i]->getBandwidthMC64();
		if (m_stats.numPartitions > m_precond_pointers[i]->getNumPartitions())
			m_stats.numPartitions = m_precond_pointers[i]->getNumPartitions();
		m_stats.time_reorder += m_precond_pointers[i]->getTimeReorder();
		m_stats.time_cpu_assemble += m_precond_pointers[i]->getTimeCPUAssemble();
		m_stats.time_transfer += m_precond_pointers[i]->getTimeTransfer();
		m_stats.time_toBanded += m_precond_pointers[i]->getTimeToBanded();
		m_stats.time_offDiags += m_precond_pointers[i]->getTimeCopyOffDiags();
		m_stats.time_bandLU += m_precond_pointers[i]->getTimeBandLU();
		m_stats.time_bandUL += m_precond_pointers[i]->getTimeBandUL();
		m_stats.time_assembly += m_precond_pointers[i]->gettimeAssembly();
		m_stats.time_fullLU += m_precond_pointers[i]->getTimeFullLU();
	}

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

	int numComponents = m_precond_pointers.size();

	if (numComponents <= 1) {
		PrecVector tmp_entries = entries;
		
		m_precond_pointers[0]->update(tmp_entries);
	}
	else {
		PrecVectorH h_entries = entries;
		
		int nnz = h_entries.size();

		std::vector<PrecVectorH> new_entries(numComponents);

		for (int i=0; i < nnz; i++)
			new_entries[m_compMap[i]].push_back(h_entries[i]);

		for (int i=0; i < numComponents; i++) {
			PrecVector tmp_entries = new_entries[i];

			m_precond_pointers[i]->update(tmp_entries);
		}
	}

	timer.Stop();

	m_stats.timeUpdate = timer.getElapsed();

	m_stats.time_reorder = 0;
	m_stats.time_cpu_assemble = m_precond_pointers[0]->getTimeCPUAssemble();
	m_stats.time_transfer = m_precond_pointers[0]->getTimeTransfer();
	m_stats.time_toBanded = m_precond_pointers[0]->getTimeToBanded();
	m_stats.time_offDiags = m_precond_pointers[0]->getTimeCopyOffDiags();
	m_stats.time_bandLU = m_precond_pointers[0]->getTimeBandLU();
	m_stats.time_bandUL = m_precond_pointers[0]->getTimeBandUL();
	m_stats.time_assembly = m_precond_pointers[0]->gettimeAssembly();
	m_stats.time_fullLU = m_precond_pointers[0]->getTimeFullLU();

	for (int i=1; i < numComponents; i++) {
		m_stats.time_cpu_assemble += m_precond_pointers[i]->getTimeCPUAssemble();
		m_stats.time_transfer += m_precond_pointers[i]->getTimeTransfer();
		m_stats.time_toBanded += m_precond_pointers[i]->getTimeToBanded();
		m_stats.time_offDiags += m_precond_pointers[i]->getTimeCopyOffDiags();
		m_stats.time_bandLU += m_precond_pointers[i]->getTimeBandLU();
		m_stats.time_bandUL += m_precond_pointers[i]->getTimeBandUL();
		m_stats.time_assembly += m_precond_pointers[i]->gettimeAssembly();
		m_stats.time_fullLU += m_precond_pointers[i]->getTimeFullLU();
	}

	return true;
}


/// Linear system solve
/**
 * This function solves the system Ax=b, for given matrix A and right-handside
 * vector b.
 *
 * An exception is throw if this call was not preceeded by a call to
 * Solver::setup().
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

	spike::bicgstab2(spmv, b_vector, x_vector, m_monitor, m_precond_pointers, m_compIndices, m_comp_perms, m_comp_reorderings);
	
	thrust::copy(x_vector.begin(), x_vector.end(), x.begin());
	timer.Stop();

	m_stats.timeSolve = timer.getElapsed();
	m_stats.residualNorm = m_monitor.getResidualNorm();
	m_stats.relResidualNorm = m_monitor.getRelResidualNorm();
	m_stats.numIterations = m_monitor.getNumIterations();

	int numComponents = m_precond_pointers.size();
	m_stats.time_shuffle = m_precond_pointers[0]->getTimeShuffle();
	for (int i=1; i < numComponents; i++)
		m_stats.time_shuffle += m_precond_pointers[i]->getTimeShuffle();

	return m_monitor.converged();
}


} // namespace spike


#endif
