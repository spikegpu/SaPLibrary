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

#include <spike/common.h>
#include <spike/components.h>
#include <spike/monitor.h>
#include <spike/precond.h>
#include <spike/bicgstab2.h>
#include <spike/spmv.h>
#include <spike/timer.h>


namespace spike {

/**
 * This structure encapsulates all solver options.
 */
struct Options
{
	Options();

	SolverType    solverType;
	int           maxNumIterations;
	double        tolerance;

	bool          performReorder;
	bool          applyScaling;
	double        dropOffFraction;

	SolverMethod  method;
	PrecondMethod precondMethod;
	bool          safeFactorization;
	bool          variableBandwidth;
	bool          singleComponent;
	bool          trackReordering;
};


/**
 * This structure encapsulates all solver statistics, both from the iterative
 * solver and the preconditioner.
 */
struct Stats
{
	Stats();

	double      timeSetup;
	double      timeUpdate;
	double      timeSolve;

	double      time_reorder;
	double      time_cpu_assemble;
	double      time_transfer;
	double      time_toBanded;
	double      time_offDiags;
	double      time_bandLU;
	double      time_bandUL;
	double      time_fullLU;
	double      time_assembly;

	double      time_shuffle;

	int         bandwidthReorder;
	int         bandwidth;

	double      actualDropOff;

	float       numIterations;
	double      residualNorm;
	double      relResidualNorm;
};


/**
 * This class encapsulates the main SPIKE::GPU solver. 
 */
template <typename Matrix, typename Vector>
class Solver
{
public:
	typedef typename Matrix::value_type   ValueType;
	typedef typename Matrix::memory_space MemorySpace;
	typedef typename cusp::coo_matrix<int, ValueType, cusp::host_memory> MatrixCOO;
	typedef typename cusp::array1d<int, cusp::host_memory> VectorI;

	Solver(int             numPartitions,
	       const Options&  opts);

	~Solver() {
		int numComponents = m_precond_pointers.size();
		for (int i=0; i<numComponents; i++)
			delete m_precond_pointers[i];
		m_precond_pointers.clear();
	}

	bool setup(const Matrix& A);

	bool update(const Vector& entries);

	template <typename SpmvOperator>
	bool solve(SpmvOperator&  spmv,
	           const Vector&  b,
	           Vector&        x);

	const Stats&  getStats() const {return m_stats;}

private:
	SolverType               m_solver;
	Monitor<Vector>          m_monitor;
	Precond<Vector>          m_precond;
	std::vector<Precond<Vector> *>  m_precond_pointers;
	std::vector<VectorI>     m_comp_reorderings;
	VectorI                  m_comp_perms;
	VectorI                  m_compIndices;
	VectorI                  m_compMap;

	int                      m_n;
	bool                     m_singleComponent;
	bool                     m_trackReordering;
	bool                     m_setupDone;

	Stats                    m_stats;
};


/**
 * This is the constructor for the Options class. It sets default values for
 * all options.
 */
Options::Options()
:	solverType(BiCGStab2),
	maxNumIterations(100),
	tolerance(1e-6),
	performReorder(true),
	applyScaling(true),
	dropOffFraction(0),
	method(LU_only),
	precondMethod(Spike),
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
	bandwidth(0),
	actualDropOff(0),
	numIterations(0),
	residualNorm(std::numeric_limits<double>::max()),
	relResidualNorm(std::numeric_limits<double>::max())
{
}


/**
 * This is the constructor for the Solver class. This constructor takes use of
 * an Options object.
 */
template <typename Matrix, typename Vector>
Solver<Matrix, Vector>::Solver(int             numPartitions,
                               const Options&  opts)
:	m_monitor(opts.maxNumIterations, opts.tolerance),
	m_precond(numPartitions, opts.performReorder, opts.applyScaling, opts.dropOffFraction, opts.method, opts.precondMethod, 
	          opts.safeFactorization, opts.variableBandwidth, opts.trackReordering),
	m_solver(opts.solverType),
	m_singleComponent(opts.singleComponent),
	m_trackReordering(opts.trackReordering),
	m_setupDone(false)
{
}

/**
 * This function performs the initial setup for the Spike solver. It prepares
 * the preconditioner based on the specified matrix A (which may be the system
 * matrix, or some approximation to it).
 */
template <typename Matrix, typename Vector>
bool
Solver<Matrix, Vector>::setup(const Matrix& A)
{
	m_n = A.num_rows;
	int nnz = A.num_entries;

	CPUTimer timer;

	timer.Start();

	MatrixCOO Acoo;
	
	Components sc(m_n);
	int numComponents;

	if (!m_singleComponent) {
		Acoo = A;
		for (int i=0; i < nnz; i++)
			sc.combineComponents(Acoo.row_indices[i], Acoo.column_indices[i]);
		sc.adjustComponentIndices();
		numComponents = sc.m_numComponents;
	} else
		numComponents = 1;

	if (m_trackReordering)
		m_compMap.resize(nnz);

	if (numComponents > 1) {
		m_compIndices = sc.m_compIndices;

		std::vector<MatrixCOO> coo_matrices(numComponents);

		m_comp_reorderings.resize(numComponents);
		if (m_comp_perms.size() != m_n)
			m_comp_perms.resize(m_n, -1);
		else
			cusp::blas::fill(m_comp_perms, -1);

		for (int i=0; i < numComponents; i++)
			m_comp_reorderings[i].clear();

		VectorI cur_indices(numComponents, 0);

		for (int i=0; i < nnz; i++) {
			int from = Acoo.row_indices[i], to = Acoo.column_indices[i];
			int compIndex = sc.m_compIndices[from];

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

			MatrixCOO &cur_matrix = coo_matrices[compIndex];
			cur_matrix.row_indices.push_back(m_comp_perms[from]);
			cur_matrix.column_indices.push_back(m_comp_perms[to]);
			cur_matrix.values.push_back(Acoo.values[i]);

			if (m_trackReordering)
				m_compMap[i] = compIndex;
		}

		for (int i=0; i < numComponents; i++) {
			MatrixCOO &cur_matrix = coo_matrices[i];
			cur_matrix.num_entries = cur_matrix.values.size();
			cur_matrix.num_rows = cur_matrix.num_cols = cur_indices[i];

			if (m_precond_pointers.size() <= i)
				m_precond_pointers.push_back(new Precond<Vector>(m_precond));

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
			m_precond_pointers.push_back(new Precond<Vector>(m_precond));

		m_precond_pointers[0]->setup(A);
	}

	timer.Stop();

	m_stats.timeSetup = timer.getElapsed();

	m_stats.bandwidthReorder = m_precond_pointers[0]->getBandwidthReordering();
	m_stats.bandwidth = m_precond_pointers[0]->getBandwidth();
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

	for (int i=1; i < numComponents; i++) {
		if (m_stats.bandwidthReorder < m_precond_pointers[i]->getBandwidthReordering())
			m_stats.bandwidthReorder = m_precond_pointers[i]->getBandwidthReordering();
		if (m_stats.bandwidth < m_precond_pointers[i]->getBandwidth())
			m_stats.bandwidth = m_precond_pointers[i]->getBandwidth();
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

/**
 * This function does an update to the banded matrix and off-diagonal matrices after
 * function setup has been called at least once and during setup, the reordering information
 * is asked to be tracked. In case at least one of the conditions is not met, errors
 * are reported.
 */
template <typename Matrix, typename Vector>
bool
Solver<Matrix, Vector>::update(const Vector& entries)
{
	// Check if this call to update() is legal.
	if (!m_setupDone) {
		fprintf(stderr, "The update function is NOT called due to the fact that the preconditioners have not been set up yet.\n");
		return false;
	}

	if (!m_trackReordering) {
		fprintf(stderr, "The update function is NOT called due to the fact that no reordering information is tracked during setup.\n");
		return false;
	}

	// Update the preconditioner.
	CPUTimer timer;
	timer.Start();


	int numComponents = m_precond_pointers.size();

	if (numComponents <= 1)
		m_precond_pointers[0] -> update(entries);
	else {
		cusp::array1d<ValueType, cusp::host_memory> h_entries = entries;
		int nnz = h_entries.size();
		std::vector<cusp::array1d<ValueType, cusp::host_memory> > new_entries(numComponents);

		for (int i=0; i < nnz; i++)
			new_entries[m_compMap[i]].push_back(h_entries[i]);

		for (int i=0; i < numComponents; i++) {
			Vector tmp_entries = new_entries[i];
			m_precond_pointers[i] -> update(tmp_entries);
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

/**
 * This function solves the system Ax=b, for given matrix A and right-handside
 * vector b.
 */
template <typename Matrix, typename Vector>
template <typename SpmvOperator>
bool
Solver<Matrix, Vector>::solve(SpmvOperator& spmv,
                              const Vector& b,
                              Vector&       x)
{
	m_monitor.init(b);

	CPUTimer timer;

	timer.Start();

	spike::bicgstab2(spmv, b, x, m_monitor, m_precond_pointers, m_compIndices, m_comp_perms, m_comp_reorderings);
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
