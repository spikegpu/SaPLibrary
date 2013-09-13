#ifndef SPIKE_SOLVER_H
#define SPIKE_SOLVER_H

#include <limits>
#include <vector>
#include <map>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/blas.h>
#include <cusp/print.h>
#include <cusp/io/matrix_market.h>

#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

#include <spike/common.h>
#include <spike/monitor.h>
#include <spike/precond.h>
#include <spike/bicgstab2.h>
#include <spike/spmv.h>
#include <spike/timer.h>


namespace spike {

// ----------------------------------------------------------------------------
// SolverComponent
//
// This structure breaks the original matrix into components and the following
// stage can deal with each component.
// ----------------------------------------------------------------------------
struct SolverComponents
{
	typedef typename cusp::array1d<int, cusp::host_memory> VectorI;

	VectorI		m_compIndices;
	int			m_n;
	int			m_numComponents;

	SolverComponents(int n);

	int			getComponentIndex(int node);
	void		combineComponents(int node1, int node2);

	void		adjustComponentIndices();
};

SolverComponents::SolverComponents(int n)
	:m_n(n)
{
	m_compIndices.resize(m_n);
	thrust::sequence(m_compIndices.begin(), m_compIndices.end());
	m_numComponents = m_n;
}

int SolverComponents::getComponentIndex(int node)
{
	if (m_compIndices[node] == node)
		return node;

	m_compIndices[node] = getComponentIndex(m_compIndices[node]);
	return m_compIndices[node];
}

void SolverComponents::combineComponents(int node1, int node2)
{
	int r1 = getComponentIndex(node1), r2 = getComponentIndex(node2);

	if (r1 != r2) {
		m_compIndices[r1] = r2;
		m_numComponents --;
	}
}

void SolverComponents::adjustComponentIndices()
{
	for (int i = 0; i < m_n; i++)
		m_compIndices[i] = getComponentIndex(i);

	std::map<int, int> compIndicesMapping;
	VectorI			   compCounts(m_numComponents, 0);

	int cur_count = 0;
	for (int i = 0; i < m_n; i++) {
		int compIndex = m_compIndices[i];
		if (compIndicesMapping.find(compIndex) == compIndicesMapping.end())
			m_compIndices[i] = compIndicesMapping[compIndex] = (++cur_count);
		else
			m_compIndices[i] = compIndicesMapping[compIndex];

		compCounts[--m_compIndices[i]]++;
	}

	int numComponents = m_numComponents;

	bool found = false;
	int selected = -1;
	for (int i = 0; i < m_numComponents; i++) {
		if (compCounts[i] == 1) {
			numComponents --;
			if (! found) {
				found = true;
				selected = i;
			}
		}
	}

	if (found) {
		m_numComponents = numComponents + 1;
		for (int i = 0; i < m_n; i++)
			if (compCounts[m_compIndices[i]] == 1)
				m_compIndices[i] = selected;

		cur_count = 0;
		compIndicesMapping.clear();
		for (int i = 0; i < m_n; i++) {
			int compIndex = m_compIndices[i];
			if (compIndicesMapping.find(compIndex) == compIndicesMapping.end())
				m_compIndices[i] = compIndicesMapping[compIndex] = (++cur_count);
			else
				m_compIndices[i] = compIndicesMapping[compIndex];

			--m_compIndices[i];
		}
	}
}


// ----------------------------------------------------------------------------
// SolverOptions
//
// This structure encapsulates all solver options.
// ----------------------------------------------------------------------------
struct SolverOptions
{
	SolverOptions();

	SolverType    solverType;
	int           maxNumIterations;
	double        tolerance;

	bool          performReorder;
	bool          applyScaling;
	double        dropOffFraction;

	SolverMethod  method;
	PrecondMethod precondMethod;
	bool          safeFactorization;
	bool          variousBandwidth;
	bool		  singleComponent;
	bool		  trackReordering;
};


// ----------------------------------------------------------------------------
// SolverStats
//
// This structure encapsulates all solver statistics, both from the iterative
// solver and the preconditioner.
// ----------------------------------------------------------------------------
struct SolverStats
{
	SolverStats();

	double      timeSetup;
	double      timeSolve;

	double		time_reorder;
	double		time_cpu_assemble;
	double		time_transfer;
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


// ----------------------------------------------------------------------------
// Solver
//
// This class encapsulates the main SPIKE::GPU solver. 
// ----------------------------------------------------------------------------
template <typename Matrix, typename Vector>
class Solver
{
public:
	typedef typename Matrix::value_type   ValueType;
	typedef typename Matrix::memory_space MemorySpace;
	typedef typename cusp::coo_matrix<int, ValueType, cusp::host_memory> MatrixCOO;
	typedef typename cusp::array1d<int, cusp::host_memory>	VectorI;

	Solver(int					numPartitions,
		   const SolverOptions&	solverOptions);

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

	const SolverStats&  getStats() const {return m_stats;}

private:
	SolverType               m_solver;
	Monitor<Vector>          m_monitor;
	Precond<Vector>          m_precond;
	std::vector<Precond<Vector> *>  m_precond_pointers;
	std::vector<VectorI>			m_comp_reorderings;
	VectorI							m_comp_perms;
	VectorI							m_compIndices;
	VectorI							m_compMap;

	int                      m_n;
	bool					 m_singleComponent;
	bool					 m_trackReordering;
	bool					 m_setupDone;

	SolverStats              m_stats;
};


// ----------------------------------------------------------------------------
// SolverOptions::SolverOptions()
//
// This is the constructor for the SolverOptions. It sets default values for
// all options.
// ----------------------------------------------------------------------------
SolverOptions::SolverOptions()
:   solverType(BiCGStab2),
	maxNumIterations(100),
	tolerance(1e-6),
	performReorder(true),
	applyScaling(true),
	dropOffFraction(0),
	method(LU_only),
	precondMethod(Spike),
	safeFactorization(false),
	variousBandwidth(true),
    singleComponent(false),
	trackReordering(true)
{
}


// ----------------------------------------------------------------------------
// SolverStats::SolverStats()
//
// This is the constructor for the SolverStats. It initializes all
// timing and performance measures.
// ----------------------------------------------------------------------------
SolverStats::SolverStats()
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

// ----------------------------------------------------------------------------
// Solver::Solver()
//
// This is the constructor for the Solver class. This constructor takes use of
// a SolverOption object.
// ----------------------------------------------------------------------------
template <typename Matrix, typename Vector>
Solver<Matrix, Vector>::Solver(int					numPartitions,
							   const SolverOptions &solverOptions)
:	m_monitor(solverOptions.maxNumIterations, solverOptions.tolerance),
	m_precond(numPartitions, solverOptions.performReorder, solverOptions.applyScaling, solverOptions.dropOffFraction, solverOptions.method, solverOptions.precondMethod, 
			  solverOptions.safeFactorization, solverOptions.variousBandwidth, solverOptions.trackReordering),
	m_solver(solverOptions.solverType),
	m_singleComponent(solverOptions.singleComponent),
	m_trackReordering(solverOptions.trackReordering),
	m_setupDone(0)
{
}

// ----------------------------------------------------------------------------
// Solver::setup()
//
// This function performs the initial setup for the Spike solver. It prepares
// the preconditioner based on the specified matrix A (which may be the system
// matrix, or some approximation to it).
// ----------------------------------------------------------------------------
template <typename Matrix, typename Vector>
bool
Solver<Matrix, Vector>::setup(const Matrix& A)
{
	m_n = A.num_rows;
	int nnz = A.num_entries;

	CPUTimer timer;

	timer.Start();

	MatrixCOO Acoo;
	
	SolverComponents sc(m_n);
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
		m_stats.actualDropOff += m_precond_pointers[i]->getActualDropOff();
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

// ----------------------------------------------------------------------------
// Solver::update()
//
// ----------------------------------------------------------------------------
template <typename Matrix, typename Vector>
bool
Solver<Matrix, Vector>::update(const Vector& entries)
{
	if (!m_setupDone) {
		fprintf(stderr, "The update function is NOT called due to the fact that the preconditioners have not been set up yet.\n");
		return false;
	}

	if (!m_trackReordering) {
		fprintf(stderr, "The update function is NOT called due to the fact that no reordering information is tracked during setup.\n");
		return false;
	}

	CPUTimer timer;
	timer.Start();

	cusp::array1d<ValueType, cusp::host_memory> h_entries = entries;

	int numComponents = m_precond_pointers.size(), nnz = h_entries.size();
	std::vector<cusp::array1d<ValueType, cusp::host_memory> > new_entries(numComponents);

	for (int i=0; i < nnz; i++)
		new_entries[m_compMap[i]].push_back(h_entries[i]);

	for (int i=0; i < numComponents; i++)
		m_precond_pointers[i] -> update(new_entries[i]);

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
		m_stats.actualDropOff += m_precond_pointers[i]->getActualDropOff();
		m_stats.time_toBanded += m_precond_pointers[i]->getTimeToBanded();
		m_stats.time_offDiags += m_precond_pointers[i]->getTimeCopyOffDiags();
		m_stats.time_bandLU += m_precond_pointers[i]->getTimeBandLU();
		m_stats.time_bandUL += m_precond_pointers[i]->getTimeBandUL();
		m_stats.time_assembly += m_precond_pointers[i]->gettimeAssembly();
		m_stats.time_fullLU += m_precond_pointers[i]->getTimeFullLU();
	}
	return true;
}

// ----------------------------------------------------------------------------
// Solver::solve()
//
// This function solves the system Ax=b, for given matrix A and right-handside
// vector b.
// ----------------------------------------------------------------------------
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
