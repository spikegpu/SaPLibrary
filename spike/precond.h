#ifndef SPIKE_PRECOND_CUH
#define SPIKE_PRECOND_CUH

#include <cusp/blas.h>
#include <cusp/print.h>

#include <spike/common.h>
#include <spike/graph.h>
#include <spike/timer.h>
#include <spike/device/factor_band_const.cuh>
#include <spike/device/factor_band_var.cuh>
#include <spike/device/sweep_band_const.cuh>
#include <spike/device/sweep_band_var.cuh>
#include <spike/device/inner_product.cuh>
#include <spike/device/shuffle.cuh>
#include <spike/device/data_transfer.cuh>

#include <omp.h>


#define MAX(A,B)   (((A) > (B)) ? (A) : (B))
#define MIN(A,B)   (((A) < (B)) ? (A) : (B))

namespace spike {


// ----------------------------------------------------------------------------
// Precond
//
// This class encapsulates the truncated Spike preconditioner.
// ----------------------------------------------------------------------------
template <typename Vector>
class Precond {
public:

	typedef typename Vector::memory_space MemorySpace;
	typedef typename Vector::value_type   ValueType;

	typedef typename cusp::array1d<int, MemorySpace> VectorI;
	typedef typename cusp::array1d<ValueType, cusp::host_memory> VectorH;
	typedef typename cusp::array1d<int, cusp::host_memory>		 MatrixMap;
	typedef typename cusp::array1d<ValueType, cusp::host_memory> MatrixMapF;
	typedef typename std::map<int64_t, int>						 IndexMap;

	Precond():m_isSetup(0) {};

	Precond(int            numPart,
	        bool           reorder,
	        bool           scale,
	        ValueType      dropOff_frac,
	        int            dropOff_k,
	        SolverMethod   method,
	        PrecondMethod  precondMethod,
	        bool           safeFactorization,
	        bool           variousBandwidth,
	        bool           secondLevelReordering,
			bool		   trackReordering);

	Precond(const		   Precond &prec);

	~Precond() {}

	Precond & operator = (const Precond &prec);

	double getTimeToBanded() const        {return m_time_toBanded;}
	double getTimeCopyOffDiags() const    {return m_time_offDiags;}
	double getTimeBandLU() const          {return m_time_bandLU;}
	double getTimeBandUL() const          {return m_time_bandUL;}
	double gettimeAssembly() const        {return m_time_assembly;}
	double getTimeFullLU() const          {return m_time_fullLU;}
	double getTimeShuffle() const         {return m_time_shuffle;}

	int    getBandwidthReordering() const {return m_k_reorder;}
	int    getBandwidth() const           {return m_k;}

	double getActualDropOff() const       {return m_dropOff_actual;}

	template <typename Matrix>
	void setup(const Matrix&  A);
	bool   isSetup() const				  {return m_isSetup;}

	void solve(Vector& v, Vector& z);

private:
	int            m_numPartitions;
	int            m_n;
	int            m_k;

	bool           m_reorder;
	bool           m_scale;
	ValueType      m_dropOff_frac;
	int            m_dropOff_k;
	SolverMethod   m_method;
	PrecondMethod  m_precondMethod;
	bool           m_safeFactorization;
	bool           m_variousBandwidth;
	bool           m_secondLevelReordering;
	bool		   m_trackReordering;

	bool		   m_isSetup;
	MatrixMap	   m_offDiagMap;
	MatrixMap	   m_WVMap;
	MatrixMap	   m_typeMap;
	MatrixMap	   m_bandedMatMap;
	MatrixMapF	   m_scaleMap;
	IndexMap	   m_idxMap;

	// Used in various-bandwidth method only, host versions
	cusp::array1d<int, cusp::host_memory>  m_ks_host;
	cusp::array1d<int, cusp::host_memory>  m_ks_row_host;
	cusp::array1d<int, cusp::host_memory>  m_ks_col_host;
	cusp::array1d<int, cusp::host_memory>  m_offDiagWidths_left_host;
	cusp::array1d<int, cusp::host_memory>  m_offDiagWidths_right_host;
	cusp::array1d<int, cusp::host_memory>  m_first_rows_host;
	cusp::array1d<int, cusp::host_memory>  m_BOffsets_host;

	// Used in various-bandwidth method only
	VectorI        m_ks;                 // All half-bandwidths
	VectorI        m_offDiagWidths_left; // All left half-bandwidths in terms of rows
	VectorI        m_offDiagWidths_right;// All right half-bandwidths in terms of rows
	VectorI		   m_offDiagPerms_left;
	VectorI		   m_offDiagPerms_right;
	VectorI        m_first_rows;
	VectorI        m_spike_ks;           // All half-bandwidths which are for spikes.
	                                     // m_spike_ks[i] = MAX ( m_ks[i] , m_ks[i+1] )
	VectorI        m_BOffsets;           // Offsets in banded-matrix B
	VectorI        m_ROffsets;           // Offsets in matrix R
	VectorI        m_WVOffsets;          // Offsets in matrix WV
	VectorI        m_compB2Offsets;      // Offsets in matrix compB2
	VectorI        m_partialBOffsets;    // Offsets in matrix partialB

	VectorI        m_optPerm;            // RCM reordering
	VectorI        m_optReordering;      // RCM reverse reordering

	VectorI        m_secondReordering;   // 2nd stage reverse reordering
	VectorI        m_secondPerm;         // 2nd stage reordering

	Vector         m_mc64RowScale;       // MC64 row scaling
	Vector         m_mc64ColScale;       // MC64 col scaling

	//// TODO: If we want to support mixed precision, the following
	//// should be of a different type (in particular, the same type
	//// as that of the matrix passed to setup()!!!)

	Vector         m_B;                  // banded matrix (LU factors)
	Vector         m_offDiags;           // contains the off-diagonal blocks of the original banded matrix
	Vector         m_R;                  // diagonal blocks in the reduced matrix (LU factors)

	VectorH        m_offDiags_host;      // Used with second-stage reorder only, copy the offDiags in SpikeGragh
	VectorH        m_WV_host;

	int            m_k_reorder;          // bandwidth after reordering

	ValueType      m_dropOff_actual;     // actual dropOff fraction achieved

	GPUTimer       m_timer;
	double         m_time_toBanded;      // GPU time for transformation or conversion to banded double       
	double         m_time_offDiags;      // GPU time to copy off-diagonal blocks
	double         m_time_bandLU;        // GPU time for LU factorization of banded blocks
	double         m_time_bandUL;        // GPU time for UL factorization of banded blocks
	double         m_time_assembly;      // GPU time for assembling the reduced matrix
	double         m_time_fullLU;        // GPU time for LU factorization of reduced matrix
	double         m_time_shuffle;       // cumulative GPU time for permutation and scaling

	template <typename Matrix>
	void transformToBandedMatrix(const Matrix&  A);

	template <typename Matrix>
	void convertToBandedMatrix(const Matrix&  A);

	template <typename Matrix>
	void updateMatrix(const Matrix&  A);

	void extractOffDiagonal(Vector& mat_WV);

	void partBandedLU();
	void partBandedLU_const();
	void partBandedLU_one();
	void partBandedLU_var();
	void partBandedUL(Vector& B);

	void partBandedFwdSweep(Vector& v);
	void partBandedFwdSweep_const(Vector& v);
	void partBandedFwdSweep_var(Vector& v);

	void partBandedBckSweep(Vector& v);
	void partBandedBckSweep_const(Vector& v);
	void partBandedBckSweep_var(Vector& v);

	void partFullLU();
	void partFullLU_const();
	void partFullLU_var();

	void partFullFwdSweep(Vector& v);
	void partFullBckSweep(Vector& v);

	void purifyRHS(Vector& v, Vector& res);

	void calculateSpikes(Vector& WV);
	void calculateSpikes_const(Vector& WV);
	void calculateSpikes_var(Vector& WV);
	void calculateSpikes_var_old(Vector& WV);

	int adjustNumThreads(int inNumThreads);

	void calculateSpikes(Vector& B2, Vector& WV);

	void assembleReducedMat(Vector& WV);

	void copyLastPartition(Vector& B2);

	void leftTrans(Vector &v, Vector &z);
	void rightTrans(Vector &v, Vector &z);

	void permute(Vector& v, VectorI& perm, Vector& w);
	void permuteAndScale(Vector& v, VectorI& perm, Vector& scale, Vector& w);
	void scaleAndPermute(Vector& v, VectorI& perm, Vector& scale, Vector& w);
	void combinePermutation(VectorI &perm, VectorI &perm2, VectorI &finalPerm);

	void getSRev(Vector& rhs, Vector& sol);
};

// ----------------------------------------------------------------------------
// Precond::Precond()
//
// This is the constructor for the Precond class.
// ----------------------------------------------------------------------------
template <typename Vector>
Precond<Vector>::Precond(int            numPart,
                         bool           reorder,
                         bool           scale,
                         ValueType      dropOff_frac,
                         int            dropOff_k,
                         SolverMethod   method,
                         PrecondMethod  precondMethod,
                         bool           safeFactorization,
                         bool           variousBandwidth,
                         bool           secondLevelReordering,
						 bool			trackReordering)
:	m_numPartitions(numPart),
	m_reorder(reorder),
	m_scale(scale),
	m_dropOff_frac(dropOff_frac),
	m_dropOff_k(dropOff_k),
	m_method(method),
	m_precondMethod(precondMethod),
	m_safeFactorization(safeFactorization),
	m_variousBandwidth(variousBandwidth),
	m_secondLevelReordering(reorder ? secondLevelReordering : false),
	m_trackReordering(trackReordering),
	m_isSetup(0),
	m_k_reorder(0),
	m_dropOff_actual(0),
	m_time_toBanded(0),
	m_time_offDiags(0),
	m_time_bandLU(0),
	m_time_bandUL(0),
	m_time_assembly(0),
	m_time_fullLU(0),
	m_time_shuffle(0)
{
}

template <typename Vector>
Precond<Vector>::Precond(const Precond<Vector> &prec):
	m_isSetup(0),
	m_k_reorder(0),
	m_dropOff_actual(0),
	m_time_toBanded(0),
	m_time_offDiags(0),
	m_time_bandLU(0),
	m_time_bandUL(0),
	m_time_assembly(0),
	m_time_fullLU(0),
	m_time_shuffle(0)
{
	m_numPartitions	=	prec.m_numPartitions;

	m_reorder		=   prec.m_reorder;;
	m_scale			=	prec.m_scale;
	m_dropOff_frac	=	prec.m_dropOff_frac;
	m_dropOff_k		=	prec.m_dropOff_k;
	m_method		=	prec.m_method;
	m_precondMethod =	prec.m_precondMethod;
	m_safeFactorization = prec.m_safeFactorization;
	m_variousBandwidth = prec.m_variousBandwidth;
	m_secondLevelReordering = prec.m_secondLevelReordering;
	m_trackReordering = prec.m_trackReordering;
}

template <typename Vector>
Precond<Vector>& Precond<Vector>::operator=(const Precond<Vector> &prec)
{
	m_numPartitions	=	prec.m_numPartitions;

	m_reorder		=   prec.m_reorder;;
	m_scale			=	prec.m_scale;
	m_dropOff_frac	=	prec.m_dropOff_frac;
	m_dropOff_k		=	prec.m_dropOff_k;
	m_method		=	prec.m_method;
	m_precondMethod =	prec.m_precondMethod;
	m_safeFactorization = prec.m_safeFactorization;
	m_variousBandwidth = prec.m_variousBandwidth;
	m_secondLevelReordering = prec.m_secondLevelReordering;
	m_trackReordering = prec.m_trackReordering;

	m_isSetup		=	0;
	m_ks_host		=	prec.m_ks_host;
	m_offDiagWidths_left_host = prec.m_offDiagWidths_left_host;
	m_offDiagWidths_right_host = prec.m_offDiagWidths_right_host;
	m_first_rows_host = prec.m_first_rows_host;
	m_BOffsets_host =   prec.m_BOffsets_host;

	m_time_shuffle = 0;
	return		*this;
}

// ----------------------------------------------------------------------------
// Precond::updateMatrix()
//
// Assume we are to solve many systems with exactly the same matrix pattern.
// When we have solved one, next time we don't bother doing permutation and 
// scaling again. Instead, we keep track of the mapping from the sparse matrix
// to the banded ones and directly update them. This function is called when
// the solver has solved at least one system.
// ----------------------------------------------------------------------------
template <typename Vector>
template <typename Matrix>
void
Precond<Vector>::updateMatrix(const Matrix& A) {
	cusp::coo_matrix<int, ValueType, cusp::host_memory> Acoo = A;
	cusp::array1d<ValueType, cusp::host_memory> B(m_B.size(), 0);
	cusp::blas::fill(m_WV_host, 0);
	cusp::blas::fill(m_offDiags_host, 0);

	int nnz = Acoo.num_entries;

	for (int i=0; i < nnz; i++) {
		int idx = m_idxMap[(int64_t)(Acoo.row_indices[i])*m_n + Acoo.column_indices[i]];

		if (m_typeMap[idx] == 2)
			m_offDiags_host[m_offDiagMap[idx]] = m_WV_host[m_WVMap[idx]] = Acoo.values[i] * m_scaleMap[idx];
		else
			B[m_bandedMatMap[idx]] = Acoo.values[i] * m_scaleMap[idx];
	}

	m_B = B;
}

// ----------------------------------------------------------------------------
// Precond::setup()
//
// This function performs the initial preconditioner setup, based on the
// specified matrix:
// (1) Reorder the matrix (MC64 and/or RCM)
// (2) Element drop-off (optional)
// (3) LU factorization
// (4) Get the reduced matrix
// ----------------------------------------------------------------------------
template <typename Vector>
template <typename Matrix>
void
Precond<Vector>::setup(const Matrix&  A)
{
	if (m_isSetup) {
		m_n = A.num_rows;
		updateMatrix(A);
	} else {
		m_n = A.num_rows;

		if (m_trackReordering)
			m_isSetup = true;

		// Form the banded matrix based on the specified matrix, either through
		// transformation (reordering and drop-off) or straight conversion.
		if (m_reorder)
			transformToBandedMatrix(A);
		else
			convertToBandedMatrix(A);
	}

	////cusp::io::write_matrix_market_file(m_B, "B.mtx");
	if (m_k == 0)
		return;


	// If we are using a single partition, perform the LU factorization
	// of the banded matrix and return.
	if (m_precondMethod == Block || m_numPartitions == 1) {

		m_timer.Start();
		partBandedLU();
		m_timer.Stop();
		m_time_bandLU = m_timer.getElapsed();

		////cusp::io::write_matrix_market_file(m_B, "B_lu.mtx");

		return;
	}
	
	// We are using more than one partition, so we must assemble the
	// truncated Spike reduced matrix R.
	m_R.resize((2 * m_k) * (2 * m_k) * (m_numPartitions - 1));

	// Extract off-diagonal blocks from the banded matrix and store them
	// in the array m_offDiags.
	Vector mat_WV;
	mat_WV.resize(2 * m_k * m_k * (m_numPartitions-1));

	m_timer.Start();
	extractOffDiagonal(mat_WV);
	m_timer.Stop();
	m_time_offDiags = m_timer.getElapsed();


	switch (m_method) {
	case LU_only:
		// In this case, we perform the partitioned LU factorization of D
		// and use the L and U factors to compute both the bottom of the 
		// right spikes (using short sweeps) and the top of the left spikes
		// (using full sweeps). Finally, we assemble the reduced matrix R.
		{
			m_timer.Start();
			partBandedLU();
			m_timer.Stop();
			m_time_bandLU = m_timer.getElapsed();

			////cusp::io::write_matrix_market_file(m_B, "B_lu.mtx");

			m_timer.Start();
			calculateSpikes(mat_WV);
			assembleReducedMat(mat_WV);
			m_timer.Stop();
			m_time_assembly = m_timer.getElapsed();
		}

		break;

	case LU_UL:
		// In this case, we perform the partitioned LU factorization of D
		// and use the L and U factors to compute the bottom of the right
		// spikes (using short sweeps).  We then perform a partitioned UL
		// factorization, using a copy of the banded matrix, and use the 
		// resulting U and L factors to compute the top of the left spikes
		// (using short sweeps). Finally, we assemble the reduced matrix R.
		{
			Vector B2 = m_B;

			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

			m_timer.Start();
			partBandedLU();
			m_timer.Stop();
			m_time_bandLU = m_timer.getElapsed();

			////cusp::io::write_matrix_market_file(m_B, "B_lu.mtx");

			m_timer.Start();
			partBandedUL(B2);
			m_timer.Stop();
			m_time_bandUL = m_timer.getElapsed();

			////cusp::io::write_matrix_market_file(B2, "B_ul.mtx");

			cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);

			m_timer.Start();
			calculateSpikes(B2, mat_WV);
			assembleReducedMat(mat_WV);
			copyLastPartition(B2);
			m_timer.Stop();
			m_time_assembly = m_timer.getElapsed();
		}

		break;
	}

	////cusp::io::write_matrix_market_file(m_B, "B_factorized.mtx");
	////cusp::io::write_matrix_market_file(mat_WV, "WV.mtx");
	////cusp::io::write_matrix_market_file(m_R, "R.mtx");

	// Perform (in-place) LU factorization of the reduced matrix.
	m_timer.Start();
	partFullLU();
	m_timer.Stop();
	m_time_fullLU = m_timer.getElapsed();

	////cusp::io::write_matrix_market_file(m_R, "R_lu.mtx");
}


// ----------------------------------------------------------------------------
// SpikePrecond::solve()
//
// This function solves the system Mz=v, for a specified vector v, where M is
// the implicitly defined preconditioner matrix.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::solve(Vector&  v,
                       Vector&  z)
{
	if (m_reorder) {
		leftTrans(v, z);
		static Vector buffer;
		buffer.resize(m_n);
		getSRev(z, buffer);
		rightTrans(buffer, z);
	} else {
		cusp::blas::copy(v, z);
		Vector buffer = z;
		getSRev(buffer, z);
	}
}


// ----------------------------------------------------------------------------
// SpikePrecond::getSRev()
//
// This function gets a rough solution of the input RHS.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::getSRev(Vector&  rhs,
                         Vector&  sol)
{
	if (m_k == 0) {
		thrust::transform(rhs.begin(), rhs.end(), m_B.begin(), sol.begin(), thrust::divides<ValueType>());
		return;
	}

	if (m_numPartitions > 1 && m_precondMethod == Spike) {
		if (!m_secondLevelReordering) {
			sol = rhs;
			// Calculate modified RHS
			partBandedFwdSweep(rhs);
			partBandedBckSweep(rhs);

			// Solve reduced system
			partFullFwdSweep(rhs);
			partFullBckSweep(rhs);

			// Purify RHS
			purifyRHS(rhs, sol);
		} else {
			static Vector buffer;
			buffer.resize(m_n);
			permute(rhs, m_secondReordering,buffer);
			// Calculate modified RHS
			partBandedFwdSweep(rhs);
			partBandedBckSweep(rhs);

			permute(rhs, m_secondReordering, sol);

			// Solve reduced system
			partFullFwdSweep(sol);
			partFullBckSweep(sol);

			purifyRHS(sol, buffer);
			permute(buffer, m_secondPerm, sol);
		}
	} else
		sol = rhs;

	// Get purified solution
	partBandedFwdSweep(sol);
	partBandedBckSweep(sol);
}


// ----------------------------------------------------------------------------
// SpikePrecond::leftTrans()
//
// This function left transforms the system. We first apply the MC64 row
// scaling and permutation (or only the MC64 row permutation) after which we
// apply the RCM row permutation.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::leftTrans(Vector&  v,
                           Vector&  z)
{
	if (m_scale)
		scaleAndPermute(v, m_optPerm, m_mc64RowScale, z);
	else
		permute(v, m_optPerm, z);
}


// ----------------------------------------------------------------------------
// SpikePrecond::rightTrans()
//
// This function right transforms the system. We apply the RCM column 
// permutation and, if needed, the MC64 column scaling.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::rightTrans(Vector&  v,
                            Vector&  z)
{
	if (m_scale)
		permuteAndScale(v, m_optReordering, m_mc64ColScale, z);
	else
		permute(v, m_optReordering, z);
}


// ----------------------------------------------------------------------------
// Precond::permute()
//
// This function transforms the input vector 'v' into the output vector 'w' by
// applying the permutation 'perm'.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::permute(Vector&   v,
                         VectorI&  perm,
                         Vector&   w)
{
	m_timer.Start();
	thrust::scatter(v.begin(), v.end(), perm.begin(), w.begin());
	m_timer.Stop();
	m_time_shuffle += m_timer.getElapsed();
}


// ----------------------------------------------------------------------------
// Precond::permuteAndScale()
//
// This function transforms the input vector 'v' into the output vector 'w' by
// applying the permutation 'perm' followed by the scaling 'scale'.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::permuteAndScale(Vector&   v,
                                 VectorI&  perm,
                                 Vector&   scale,
                                 Vector&   w)
{
	m_timer.Start();

	thrust::scatter(
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.begin(), thrust::make_permutation_iterator(scale.begin(), perm.begin()))), Multiplier<ValueType>()),
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.end(), thrust::make_permutation_iterator(scale.end(), perm.end()))), Multiplier<ValueType>()),
			perm.begin(),
			w.begin()
			);

	m_timer.Stop();
	m_time_shuffle += m_timer.getElapsed();
}


// ----------------------------------------------------------------------------
// Precond::scaleAndPermute()
//
// This function transforms the input vector 'v' into the output vector 'w' by
// applying the scaling 'scale' followed by the permutation 'perm'.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::scaleAndPermute(Vector&   v,
                                 VectorI&  perm,
                                 Vector&   scale,
                                 Vector&   w)
{
	m_timer.Start();
	thrust::scatter(
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.begin(), scale.begin())), Multiplier<ValueType>()),
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.end(), scale.end())), Multiplier<ValueType>()),
			perm.begin(),
			w.begin()
			);
	m_timer.Stop();
	m_time_shuffle += m_timer.getElapsed();
}


// ----------------------------------------------------------------------------
// Precond::combinePermutation()
//
// This function combines two permutations to one.
// ----------------------------------------------------------------------------
template <typename Vector>
void Precond<Vector>::combinePermutation(VectorI&  perm,
                                         VectorI&  perm2,
                                         VectorI&  finalPerm)
{
	m_timer.Start();
	thrust::gather(perm.begin(), perm.end(), perm2.begin(), finalPerm.begin());
	m_timer.Stop();
	m_time_shuffle += m_timer.getElapsed();
}


// ----------------------------------------------------------------------------
// Precond::transformToBandedMatrix()
//
// This function applies the reordering and element drop-off algorithms to
// obtain the banded matrix for the Spike method. On return, the following
// member variables are set:
//   m_B
//       banded matrix after reordering and drop-off. This matrix is stored
//       column-wise, band after band, in a contiguous 1-D array.
//   m_k
//       half band-width of the matrix m_B (after reordering and drop-off)
//   m_optReordering
//   m_optPerm
//       permutation arrays obtained from the symmetric RCM algorithm
//       row and column permutations obtained from the MC64 algorithm
//   mc64RowScale
//   mc64ColScale
//       row and column scaling factors obtained from the MC64 algorithm
// ----------------------------------------------------------------------------
template <typename Vector>
template <typename Matrix>
void
Precond<Vector>::transformToBandedMatrix(const Matrix&  A)
{
	// Reorder the matrix and apply drop-off. For this, we convert the
	// input matrix to COO format and copy it on the host.
	cusp::coo_matrix<int, ValueType, cusp::host_memory> Acoo = A;

	cusp::array1d<ValueType, cusp::host_memory>  B;
	cusp::array1d<int, cusp::host_memory>        optReordering;
	cusp::array1d<int, cusp::host_memory>        optPerm;
	cusp::array1d<int, cusp::host_memory>        mc64RowPerm;
	cusp::array1d<ValueType, cusp::host_memory>  mc64RowScale;
	cusp::array1d<ValueType, cusp::host_memory>  mc64ColScale;
	cusp::array1d<int, cusp::host_memory>        secondReorder;
	cusp::array1d<int, cusp::host_memory>        secondPerm;

	SpikeGraph<ValueType>						 graph(m_trackReordering);

	cusp::array1d<int, cusp::host_memory>		 offDiagPerms_left;
	cusp::array1d<int, cusp::host_memory>		 offDiagPerms_right;

	const ValueType dropMin = 1.0/100;

	double time_reorder = 0, time_assemble = 0, time_transfer = 0;
	CPUTimer reorder_timer, assemble_timer, transfer_timer;

	reorder_timer.Start();
	m_k_reorder = graph.reorder(Acoo, m_scale, optReordering, optPerm, mc64RowPerm, mc64RowScale, mc64ColScale, m_idxMap, m_scaleMap);
	reorder_timer.Stop();

	time_reorder += reorder_timer.getElapsed();
	
	int dropped = 0;

	if (m_dropOff_frac > 0) {
		if (!m_secondLevelReordering)
			dropped = graph.dropOff(m_dropOff_frac, m_dropOff_actual);
		else
			dropped = graph.dropOff(m_dropOff_frac, m_dropOff_actual, dropMin);
	}
	else if (m_dropOff_k > 0)
		dropped = graph.dropOff(m_dropOff_k, m_dropOff_actual);
	else
		m_dropOff_actual = 0;

	// FIXME: this is a little bit problematic when for some off-diagonals, there is no element at all.
	m_k = m_k_reorder - dropped;


	// Verify that the required number of partitions is consistent with the
	// problem size and half-bandwidth.  If 'n' is the smallest partition size,
	// the following condition must be satisfied:
	//   K+1 <= n   (for Spike algorithm)
	// These imply a maximum allowable number of partitions.
	int maxNumPartitions = MAX(m_n / (m_k + 1), 1);

	if (m_numPartitions > maxNumPartitions) {
		std::cerr << "P = " << m_numPartitions << " is too large for N = "
			<< m_n << " and K = " << m_k << std::endl
			<< "The number of partitions was reset to P = " << maxNumPartitions << std::endl;
		m_numPartitions = maxNumPartitions;
	}

	// If there is just one partition, do not use variable band and
	// second stage reordering.
	if (m_numPartitions == 1 || m_k == 0) {
		if (m_variousBandwidth)
			std::cerr << "A single partition is used or the half-bandwidth is zero. Variable-band option was disabled." << std::endl;
		m_variousBandwidth = false;
		m_secondLevelReordering = false;
	}

	if (m_dropOff_frac > 0) {
		if (m_variousBandwidth)
			graph.dropOffPost(m_dropOff_frac, m_dropOff_actual, dropMin, m_numPartitions);
	}

	// Assemble the banded matrix.
	if (m_variousBandwidth) {
		assemble_timer.Start();
		graph.assembleOffDiagMatrices(m_k, m_numPartitions, m_WV_host, m_offDiags_host, m_offDiagWidths_left_host, m_offDiagWidths_right_host, offDiagPerms_left, offDiagPerms_right, m_typeMap, m_offDiagMap, m_WVMap);
		assemble_timer.Stop();
		time_assemble += assemble_timer.getElapsed();

		reorder_timer.Start();
		graph.secondLevelReordering(m_k, m_numPartitions, secondReorder, secondPerm, m_first_rows_host);
		reorder_timer.Stop();

		assemble_timer.Start();
		time_reorder += reorder_timer.getElapsed();
		graph.assembleBandedMatrix(m_k, m_numPartitions, m_ks_col_host, m_ks_row_host, B,
		                           m_ks_host, m_BOffsets_host, 
								   m_typeMap, m_bandedMatMap);
		assemble_timer.Stop();
		time_assemble += assemble_timer.getElapsed();
	} else {
		assemble_timer.Start();
		graph.assembleBandedMatrix(m_k, m_ks_col_host, m_ks_row_host, B, m_typeMap, m_bandedMatMap);
		assemble_timer.Stop();
		time_assemble += assemble_timer.getElapsed();
	}

	transfer_timer.Start();

	// Copy the banded matrix and permutation data to the device.
	m_optReordering = optReordering;
	m_optPerm = optPerm;

	if (m_scale) {
		m_mc64RowScale = mc64RowScale;
		m_mc64ColScale = mc64ColScale;
	}
	m_B = B;

	if (m_variousBandwidth) {
		m_ks = m_ks_host;
		m_offDiagWidths_left = m_offDiagWidths_left_host;
		m_offDiagWidths_right = m_offDiagWidths_right_host;
		m_offDiagPerms_left = offDiagPerms_left;
		m_offDiagPerms_right = offDiagPerms_right;
		m_BOffsets = m_BOffsets_host;

		m_spike_ks.resize(m_numPartitions - 1);
		m_ROffsets.resize(m_numPartitions - 1);
		m_WVOffsets.resize(m_numPartitions - 1);
		m_compB2Offsets.resize(m_numPartitions - 1);
		m_partialBOffsets.resize(m_numPartitions-1);

		cusp::blas::fill(m_spike_ks, m_k);
		thrust::sequence(m_ROffsets.begin(), m_ROffsets.end(), 0, 4*m_k*m_k);
		thrust::sequence(m_WVOffsets.begin(), m_WVOffsets.end(), 0, 2*m_k*m_k);
		thrust::sequence(m_partialBOffsets.begin(), m_partialBOffsets.end(), 0, 2 * (m_k+1) * (2*m_k+ 1));
		thrust::sequence(m_compB2Offsets.begin(), m_compB2Offsets.end(), 0, 2 * m_k * (2 * m_k + 1));
	}

	if (m_secondLevelReordering) {
		VectorI buffer2(m_n);
		m_secondReordering = secondReorder;
		combinePermutation(m_secondReordering, m_optReordering, buffer2);
		m_optReordering = buffer2;

		m_secondPerm = secondPerm;
		combinePermutation(m_optPerm, m_secondPerm, buffer2);
		m_optPerm = buffer2;

		m_first_rows = m_first_rows_host;
	}

	{
		VectorI buffer = mc64RowPerm, buffer2(m_n);
		combinePermutation(buffer, m_optPerm, buffer2);
		m_optPerm = buffer2;
	}

	transfer_timer.Stop();
	time_transfer += transfer_timer.getElapsed();

	fprintf(stderr, "Reorder time: %g, CPU assemble time: %g, data transfer time: %g\n", time_reorder, time_assemble, time_transfer);
}


// ----------------------------------------------------------------------------
// Precond::convertToBandedMatrix()
//
// This function converts the specified sparse format matrix to a banded matrix
// m_B which is stored column-wise, band after band, in a contiguous 1-D array.
// It also sets m_k to be the half-bandwidth of m_B.
// ----------------------------------------------------------------------------
template <typename Vector>
template <typename Matrix>
void
Precond<Vector>::convertToBandedMatrix(const Matrix&  A)
{
/*
	cusp::coo_matrix<int, ValueType, cusp::host_memory> Acoo = A;
	cusp::array1d<ValueType, cusp::host_memory>         B;
	SpikeGraph<ValueType>  graph;
	m_k = graph.convert(Acoo, B);
	m_B = B;
*/


	// Convert matrix to COO format.
	cusp::coo_matrix<int, ValueType, MemorySpace> Acoo = A;
	int n = Acoo.num_rows;
	int nnz = Acoo.num_entries;

	// Calculate bandwidth. Note that we use an explicit code block so
	// that the temporary array 'buffer' is freed before we resize m_B
	// (otherwise, we may run out of memory).
	{
		VectorI  buffer(nnz);
		cusp::blas::axpby(Acoo.row_indices, Acoo.column_indices, buffer, 1, -1);
		m_k = cusp::blas::nrmmax(buffer);
	}

	// Verify that the required number of partitions is consistent with the
	// problem size and half-bandwidth.  If 'n' is the smallest partition size,
	// the following two conditions must be satisfied:
	//   (1)  K+1 <= n   (for Spike algorithm)
	//   (2)  2*K <= n   (for current implementation of UL)
	// These imply a maximum allowable number of partitions.
	int  maxNumPartitions = MAX(1, m_n / MAX(m_k + 1, 2 * m_k));
	if (m_numPartitions > maxNumPartitions) {
		std::cerr << "P = " << m_numPartitions << " is too large for N = "
			<< m_n << " and K = " << m_k << std::endl
			<< "The number of partitions was reset to P = " << maxNumPartitions << std::endl;
		m_numPartitions = maxNumPartitions;
	}

	// If there is just one partition, it's meaningless to apply various-bandwidth method
	if (m_numPartitions == 1) {
		if (m_variousBandwidth) {
			std::cerr << "Partition number equals one, it's thus meaningless to use various-bandwidth method." << std::endl;
			m_variousBandwidth = false;
		}
	}

	// Set the size and load the banded matrix into m_B.
	m_B.resize((2*m_k+1)*n);

	int blockX = nnz, gridX = 1, gridY = 1;
	kernelConfigAdjust(blockX, gridX, gridY, BLOCK_SIZE, MAX_GRID_DIMENSION);
	dim3 grids(gridX, gridY);

	int *d_rows = thrust::raw_pointer_cast(&(Acoo.row_indices[0]));
	int *d_cols = thrust::raw_pointer_cast(&(Acoo.column_indices[0]));
	ValueType *d_vals = thrust::raw_pointer_cast(&(Acoo.values[0]));
	ValueType *dB = thrust::raw_pointer_cast(&m_B[0]);

	m_timer.Start();
	device::copyFromCOOMatrixToBandedMatrix<<<grids, blockX>>>(nnz, m_k, d_rows, d_cols, d_vals, dB);
	m_timer.Stop();
	m_time_toBanded = m_timer.getElapsed();
}



// ----------------------------------------------------------------------------
// Precond::extractOffDiagonal()
//
// This function extracts and saves the off-diagonal blocks. Simultaneously,
// it also initializes the specified WV matrix with the off-diagonal blocks
// (this will be later processed to obtain the actual spike blocks in WV).
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::extractOffDiagonal(Vector&  mat_WV)
{
	// If second-level reordering is enabled, the off-diagonal matrices are already in the host.
	if (m_secondLevelReordering) {
		mat_WV = m_WV_host;
		m_offDiags = m_offDiags_host;
		return;
	}

	m_offDiags.resize(2 * m_k * m_k * (m_numPartitions - 1));

	ValueType* p_B = thrust::raw_pointer_cast(&m_B[0]);
	ValueType* p_WV = thrust::raw_pointer_cast(&mat_WV[0]);
	ValueType* p_offDiags = thrust::raw_pointer_cast(&m_offDiags[0]);
	int* p_ks = thrust::raw_pointer_cast(&m_ks[0]);

	int  partSize  = m_n / m_numPartitions;
	int  remainder = m_n % m_numPartitions;

	dim3 grids(m_k, m_numPartitions-1);

	if (m_k > 1024)
		device::copydWV_general<ValueType><<<grids, 512>>>(m_k, p_B, p_WV, p_offDiags, partSize, m_numPartitions, remainder);
	else if (m_k > 32)
		device::copydWV_g32<ValueType><<<grids, m_k>>>(m_k, p_B, p_WV, p_offDiags, partSize, m_numPartitions, remainder);
	else
		device::copydWV<ValueType><<<m_numPartitions-1, m_k*m_k>>>(m_k, p_B, p_WV, p_offDiags, partSize, m_numPartitions, remainder);
}



// ----------------------------------------------------------------------------
// Precond::partFullLU()
// Precond::partFullLU_const()
// Precond::partFullLU_var()
//
// This function performs the in-place LU factorization of the diagonal blocks
// of the reduced matrix R. We take advantage of the special block structure of
// each individual 2*k by 2*k diagonal block, namely:
//       [ I_k  |   V  ]
// R_i = [------+ -----]
//       [  W   |  I_k ]
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::partFullLU()
{
	if (!m_variousBandwidth)
		partFullLU_const();
	else
		partFullLU_var();
}

template <typename Vector>
void
Precond<Vector>::partFullLU_const()
{
	ValueType* d_R = thrust::raw_pointer_cast(&m_R[0]);
	int        two_k = 2 * m_k;

	// The first k rows of each diagonal block do not need a division step and
	// always use a pivot = 1.
	{
		dim3 grids(m_k, m_numPartitions-1);

		if( m_k > 1024)
			device::fullLU_sub_spec_general<ValueType><<<grids, 512>>>(d_R, two_k, m_k);
		else
			device::fullLU_sub_spec<ValueType><<<grids, m_k>>>(d_R, two_k, m_k);
	}

	// The following k rows of each diagonal block require first a division by
	// the pivot.
	if (m_safeFactorization) {
		for(int i = m_k; i < two_k-1; i++) {
			int  threads = two_k-1-i;
			dim3 grids(two_k-1-i, m_numPartitions-1);

			if(threads > 1024) {
				device::fullLU_div_safe_general<ValueType><<<m_numPartitions-1, 512>>>(d_R, m_k, two_k, i);
				device::fullLU_sub_general<ValueType><<<grids, 512>>>(d_R, m_k, two_k, i);
			} else {
				device::fullLU_div_safe<ValueType><<<m_numPartitions-1, threads>>>(d_R, two_k, i);
				device::fullLU_sub<ValueType><<<grids, threads>>>(d_R, two_k, i);
			}
		}
	} else {
		for(int i = m_k; i < two_k-1; i++) {
			int  threads = two_k-1-i;
			dim3 grids(two_k-1-i, m_numPartitions-1);

			if(threads > 1024) {
				device::fullLU_div_general<ValueType><<<m_numPartitions-1, 512>>>(d_R, m_k, two_k, i);
				device::fullLU_sub_general<ValueType><<<grids, 512>>>(d_R, m_k, two_k, i);
			} else {
				device::fullLU_div<ValueType><<<m_numPartitions-1, threads>>>(d_R, two_k, i);
				device::fullLU_sub<ValueType><<<grids, threads>>>(d_R, two_k, i);
			}
		}
	}
}

template <typename Vector>
void
Precond<Vector>::partFullLU_var()
{
	ValueType* d_R = thrust::raw_pointer_cast(&m_R[0]);
	int*	   p_spike_ks = thrust::raw_pointer_cast(&m_spike_ks[0]);
	int*	   p_ROffsets = thrust::raw_pointer_cast(&m_ROffsets[0]);
	int        two_k = 2 * m_k;

	// The first k rows of each diagonal block do not need a division step and
	// always use a pivot = 1.
	{
		dim3 grids(m_k, m_numPartitions-1);

		if( m_k > 1024)
			device::var::fullLU_sub_spec_general<ValueType><<<grids, 512>>>(d_R, p_spike_ks, p_ROffsets);
		else
			device::var::fullLU_sub_spec<ValueType><<<grids, m_k>>>(d_R, p_spike_ks, p_ROffsets);
	}

	// The following k rows of each diagonal block require first a division by
	// the pivot.
	if (m_safeFactorization) {
		for(int i = m_k; i < two_k-1; i++) {
			int  threads = two_k-1-i;
			dim3 grids(two_k-1-i, m_numPartitions-1);

			if(threads > 1024) {
				device::var::fullLU_div_safe_general<ValueType><<<m_numPartitions-1, 512>>>(d_R, p_spike_ks, p_ROffsets, i);
				device::var::fullLU_sub_general<ValueType><<<grids, 512>>>(d_R, p_spike_ks, p_ROffsets, i);
			} else {
				device::var::fullLU_div_safe<ValueType><<<m_numPartitions-1, threads>>>(d_R, p_spike_ks, p_ROffsets, i);
				device::var::fullLU_sub<ValueType><<<grids, threads>>>(d_R, p_spike_ks, p_ROffsets, i);
			}
		}
	} else {
		for(int i = m_k; i < two_k-1; i++) {
			int  threads = two_k-1-i;
			dim3 grids(two_k-1-i, m_numPartitions-1);

			if(threads > 1024) {
				device::var::fullLU_div_general<ValueType><<<m_numPartitions-1, 512>>>(d_R, p_spike_ks,  p_ROffsets, i);
				device::var::fullLU_sub_general<ValueType><<<grids, 512>>>(d_R, p_spike_ks,  p_ROffsets, i);
			} else {
				device::var::fullLU_div<ValueType><<<m_numPartitions-1, threads>>>(d_R, p_spike_ks,  p_ROffsets, i);
				device::var::fullLU_sub<ValueType><<<grids, threads>>>(d_R, p_spike_ks,  p_ROffsets, i);
			}
		}
	}

	{
		dim3 grids(m_k-1, m_numPartitions-1);
		if (m_k >= 1024)
			device::var::fullLU_post_divide_general<ValueType><<<grids, 512>>>(d_R, p_spike_ks, p_ROffsets);
		else
			device::var::fullLU_post_divide<ValueType><<<grids, m_k-1>>>(d_R, p_spike_ks, p_ROffsets);
	}
}


// ----------------------------------------------------------------------------
// Precond::partBandedLU()
// Precond::partBandedLU_one()
// Precond::partBandedLU_const()
// Precond::partBandedLU_var()
//
// This function performs the in-place LU factorization of the diagonal blocks
// of the specified banded matrix B, on a per-partition basis, using the
// "window sliding" method.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::partBandedLU()
{
	if (!m_variousBandwidth) {
		if (m_numPartitions > 1)
			partBandedLU_const();
		else
			partBandedLU_one();
	}
	else
		partBandedLU_var();
}

template <typename Vector>
void
Precond<Vector>::partBandedLU_one()
{
	ValueType* dB = thrust::raw_pointer_cast(&m_B[0]);

	if(m_k >= CRITICAL_THRESHOLD) {
		int threadsNum = 0;

		for (int st_row = 0; st_row < m_n-1; st_row++) {
			threadsNum = m_ks_col_host[st_row];
			// if (threadsNum > m_n - st_row - 1)
				// threadsNum = m_n - st_row - 1;
			int blockX = m_ks_row_host[st_row];
			// if (blockX > m_n - st_row - 1)
				// blockX = m_n - st_row - 1;
			if (threadsNum > 1024) {
				if (m_safeFactorization)
					device::bandLU_critical_div_onePart_safe_general<ValueType><<<1, 512>>>(dB, st_row, m_k, threadsNum);
				else
					device::bandLU_critical_div_onePart_general<ValueType><<<threadsNum/512+1, 512>>>(dB, st_row, m_k, threadsNum);

				device::bandLU_critical_sub_onePart_general<ValueType><<<blockX, 512>>>(dB, st_row, m_k, threadsNum);
			} else {
				if (m_safeFactorization)
					device::bandLU_critical_div_onePart_safe<ValueType><<<1, threadsNum>>>(dB, st_row, m_k);
				else
					device::bandLU_critical_div_onePart<ValueType><<<1, threadsNum>>>(dB, st_row, m_k);

				device::bandLU_critical_sub_onePart<ValueType><<<blockX, threadsNum>>>(dB, st_row, m_k);
			}
		}
	} else if (m_k > 27) {
		if (m_safeFactorization)
			device::bandLU_g32_safe<ValueType><<<1, 512>>>(dB, m_k, m_n, 0);
		else
			device::bandLU_g32<ValueType><<<1, 512>>>(dB, m_k, m_n, 0);
	} else {
		if (m_safeFactorization)
			device::bandLU_safe<ValueType><<<1,  m_k * m_k>>>(dB, m_k, m_n, 0);
		else
			device::bandLU<ValueType><<<1,  m_k * m_k>>>(dB, m_k, m_n, 0);
			////device::swBandLU<ValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
	}

	int gridX = m_n, gridY = 1;
	kernelConfigAdjust(gridX, gridY, MAX_GRID_DIMENSION);
	dim3 grids(gridX, gridY);
	if (m_k > 1024)
		device::bandLU_post_divide_general<ValueType><<<grids, 512>>>(dB, m_k, m_n);
	else
		device::bandLU_post_divide<ValueType><<<grids, m_k>>>(dB, m_k, m_n);
}

template <typename Vector>
void
Precond<Vector>::partBandedLU_const()
{
	ValueType* dB = thrust::raw_pointer_cast(&m_B[0]);

	int n_eff = m_n;
	int numPart_eff = m_numPartitions;

	if (m_method == LU_UL && m_numPartitions > 1 && m_precondMethod != Block) {
		n_eff -= m_n / m_numPartitions;
		numPart_eff--;
	}

	int partSize  = n_eff / numPart_eff;
	int remainder = n_eff % numPart_eff;

	if(m_k >= CRITICAL_THRESHOLD) {
		int final_partition_size = partSize + 1;
		int threadsNum = 0;

		for (int st_row = 0; st_row < final_partition_size - 1; st_row++) {
			if (st_row == 0) {
				if (remainder == 0) continue;
				threadsNum = m_k;
				if (threadsNum > final_partition_size - 1)
					threadsNum = final_partition_size - 1;
				if (threadsNum > 1024) {
					if (m_safeFactorization)
						device::bandLU_critical_div_safe_general<ValueType><<<remainder, 512>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandLU_critical_div_general<ValueType><<<remainder, 512>>>(dB, st_row, m_k, partSize, remainder);
					dim3 tmpGrids(threadsNum, remainder);
					device::bandLU_critical_sub_general<ValueType><<<tmpGrids, 512>>>(dB, st_row, m_k, partSize, remainder);
				} else {
					if (m_safeFactorization)
						device::bandLU_critical_div_safe<ValueType><<<remainder, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandLU_critical_div<ValueType><<<remainder, threadsNum>>>(dB, st_row, m_k, partSize, remainder);

					dim3 tmpGrids(threadsNum, remainder);
					device::bandLU_critical_sub<ValueType><<<tmpGrids, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
				}
			} else {
				threadsNum = m_k;
				if (threadsNum > final_partition_size - st_row - 1)
					threadsNum = final_partition_size - st_row - 1;
				if (threadsNum > 1024) {
					if (m_safeFactorization)
						device::bandLU_critical_div_safe_general<ValueType><<<numPart_eff, 512>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandLU_critical_div_general<ValueType><<<numPart_eff, 512>>>(dB, st_row, m_k, partSize, remainder);

					dim3 tmpGrids(threadsNum, numPart_eff);
					device::bandLU_critical_sub_general<ValueType><<<tmpGrids, 512>>>(dB, st_row, m_k, partSize, remainder);
				} else {
					if (m_safeFactorization)
						device::bandLU_critical_div_safe<ValueType><<<numPart_eff, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandLU_critical_div<ValueType><<<numPart_eff, threadsNum>>>(dB, st_row, m_k, partSize, remainder);

					dim3 tmpGrids(threadsNum, numPart_eff);
					device::bandLU_critical_sub<ValueType><<<tmpGrids, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
				}
			}
		}
	} else if (m_k > 27) {
		if (m_safeFactorization)
			device::bandLU_g32_safe<ValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder);
		else
			device::bandLU_g32<ValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder);
	} else {
		if (m_safeFactorization)
			device::bandLU_safe<ValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
		else
			device::bandLU<ValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
			////device::swBandLU<ValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
	}

	if (m_numPartitions == 1) {
		int gridX = m_n, gridY = 1;
		kernelConfigAdjust(gridX, gridY, MAX_GRID_DIMENSION);
		dim3 grids(gridX, gridY);
		if (m_k > 1024)
			device::bandLU_post_divide_general<ValueType><<<grids, 512>>>(dB, m_k, m_n);
		else
			device::bandLU_post_divide<ValueType><<<grids, m_k>>>(dB, m_k, m_n);
	}
}


template <typename Vector>
void
Precond<Vector>::partBandedLU_var()
{
	ValueType* dB = thrust::raw_pointer_cast(&m_B[0]);
	int *p_ks = thrust::raw_pointer_cast(&m_ks[0]);
	int tmp_k = cusp::blas::nrmmax(m_ks);
	int *p_BOffsets = thrust::raw_pointer_cast(&m_BOffsets[0]);

	int partSize  = m_n / m_numPartitions;
	int remainder = m_n % m_numPartitions;

	if(tmp_k >= CRITICAL_THRESHOLD) {
		int final_partition_size = partSize + 1;
		int blockY = 0, threadsNum = adjustNumThreads(cusp::blas::nrm1(m_ks) / m_numPartitions), last = 0;

		for (int st_row = 0; st_row < final_partition_size - 1; st_row++) {
			if (st_row == 0) {
				if (remainder == 0) continue;

				blockY = m_ks_row_host[st_row];
				last = m_ks_col_host[st_row];
				int corres_row = st_row;
				for (int i = 1; i < remainder; i++) {
					corres_row += partSize + 1;
					if (blockY < m_ks_row_host[corres_row])
						blockY = m_ks_row_host[corres_row];
					if (last < m_ks_col_host[corres_row])
						last = m_ks_col_host[corres_row];
				}

				if (m_safeFactorization)
					device::var::bandLU_critical_div_safe_general<ValueType><<<remainder, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder);
				else
					device::var::bandLU_critical_div_general<ValueType><<<remainder, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder);

				dim3 tmpGrids(blockY, remainder);
				device::var::bandLU_critical_sub_general<ValueType><<<tmpGrids, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder, last);
			} else {
				blockY = 0;
				last = 0;
				int corres_row = st_row;
				for (int i = 0; i < remainder; i++) {
					if (blockY < m_ks_row_host[corres_row])
						blockY = m_ks_row_host[corres_row];
					if (last < m_ks_col_host[corres_row])
						last = m_ks_col_host[corres_row];
					corres_row += partSize + 1;
				}
				corres_row --;
				for (int i = remainder; i < m_numPartitions; i++) {
					if (blockY < m_ks_row_host[corres_row])
						blockY = m_ks_row_host[corres_row];
					if (last < m_ks_col_host[corres_row])
						last = m_ks_col_host[corres_row];
					corres_row += partSize;
				}

				if (m_safeFactorization)
					device::var::bandLU_critical_div_safe_general<ValueType><<<m_numPartitions, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder);
				else
					device::var::bandLU_critical_div_general<ValueType><<<m_numPartitions, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder);

				dim3 tmpGrids(blockY, m_numPartitions);
				device::var::bandLU_critical_sub_general<ValueType><<<tmpGrids, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder, last);
			}
		}
	} else if (tmp_k > 27){
		if (m_safeFactorization)
			device::var::bandLU_g32_safe<ValueType><<<m_numPartitions, 512>>>(dB, p_ks, p_BOffsets, partSize, remainder);
		else
			device::var::bandLU_g32<ValueType><<<m_numPartitions, 512>>>(dB, p_ks, p_BOffsets, partSize, remainder);
	} else {
		if (m_safeFactorization)
			device::var::bandLU_safe<ValueType><<<m_numPartitions,  tmp_k * tmp_k >>>(dB, p_ks, p_BOffsets, partSize, remainder);
		else
			device::var::bandLU<ValueType><<<m_numPartitions,  tmp_k * tmp_k>>>(dB, p_ks, p_BOffsets, partSize, remainder);
	}

	int gridX = partSize+1, gridY = 1;
	kernelConfigAdjust(gridX, gridY, MAX_GRID_DIMENSION);
	dim3 grids(gridX, gridY);

	for (int i=0; i<m_numPartitions ; i++) {
		if (i < remainder) {
			if (m_ks_host[i] <= 1024)
				device::var::bandLU_post_divide_per_partition<ValueType><<<grids, m_ks_host[i]>>>(dB, m_ks_host[i], m_BOffsets_host[i], partSize + 1);
			else
				device::var::bandLU_post_divide_per_partition_general<ValueType><<<grids, 512>>>(dB, m_ks_host[i], m_BOffsets_host[i], partSize + 1);
		}
		else {
			if (m_ks_host[i] <= 1024)
				device::var::bandLU_post_divide_per_partition<ValueType><<<grids, m_ks_host[i]>>>(dB, m_ks_host[i], m_BOffsets_host[i], partSize);
			else
				device::var::bandLU_post_divide_per_partition_general<ValueType><<<grids, 512>>>(dB, m_ks_host[i], m_BOffsets_host[i], partSize);
		}
	}
}


// ----------------------------------------------------------------------------
// Precond::partBandedUL()
//
// This function performs the in-place UL factorization of the diagonal blocks
// of the specified banded matrix B, on a per-partition basis, using the
// "window sliding" method.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::partBandedUL(Vector& B)
{
	int partSize  = m_n / m_numPartitions;
	int remainder = m_n % m_numPartitions;

	ValueType* dB = thrust::raw_pointer_cast(&B[(2 * m_k + 1) * (remainder == 0 ? partSize : (partSize + 1))]);

	int n_eff = m_n - (remainder == 0 ? partSize : (partSize+1));
	int numPart_eff = m_numPartitions - 1;

	partSize = n_eff / numPart_eff;
	remainder = n_eff % numPart_eff;

	if(m_k >= CRITICAL_THRESHOLD) {
		int final_partition_size = partSize + 1;
		int threadsNum = 0;
		for (int st_row = final_partition_size - 1; st_row > 0; st_row--) {
			if (st_row == final_partition_size - 1) {
				if (remainder == 0) continue;
				threadsNum = m_k;
				if(st_row < m_k)
					threadsNum = st_row;
				dim3 tmpGrids(threadsNum, remainder);
				if (threadsNum > 1024) {
					if (m_safeFactorization)
						device::bandUL_critical_div_safe_general<ValueType><<<remainder, 512>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandUL_critical_div_general<ValueType><<<remainder, 512>>>(dB, st_row, m_k, partSize, remainder);
					device::bandUL_critical_sub_general<ValueType><<<tmpGrids, 512>>>(dB, st_row, m_k, partSize, remainder);
				} else {
					if (m_safeFactorization)
						device::bandUL_critical_div_safe<ValueType><<<remainder, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandUL_critical_div<ValueType><<<remainder, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					device::bandUL_critical_sub<ValueType><<<tmpGrids, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
				}
			} else {
				threadsNum = m_k;
				if(st_row < m_k)
					threadsNum = st_row;
				dim3 tmpGrids(threadsNum, numPart_eff);
				if(threadsNum > 1024) {
					if (m_safeFactorization)
						device::bandUL_critical_div_safe_general<ValueType> <<<numPart_eff, 512>>> (dB, st_row, m_k, partSize, remainder);
					else
						device::bandUL_critical_div_general<ValueType> <<<numPart_eff, 512>>> (dB, st_row, m_k, partSize, remainder);
					device::bandUL_critical_sub_general<ValueType> <<<tmpGrids, 512>>> (dB, st_row, m_k, partSize, remainder);
				} else {
					if (m_safeFactorization)
						device::bandUL_critical_div_safe<ValueType> <<<numPart_eff, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandUL_critical_div<ValueType> <<<numPart_eff, threadsNum>>> (dB, st_row, m_k, partSize, remainder);
					device::bandUL_critical_sub<ValueType> <<<tmpGrids, threadsNum>>> (dB, st_row, m_k, partSize, remainder);
				}
			}
		}
	} else if (m_k > 27) {
		if (m_safeFactorization)
			device::bandUL_g32_safe<ValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder);
		else
			device::bandUL_g32<ValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder);
	} else {
		if (m_safeFactorization)
			device::bandUL_safe<ValueType><<<numPart_eff, m_k * m_k>>>(dB, m_k, partSize, remainder);
		else
			device::bandUL<ValueType><<<numPart_eff, m_k * m_k>>>(dB, m_k, partSize, remainder);
			////device::swBandUL<ValueType><<<numPart_eff, m_k * m_k>>>(dB, m_k, partSize, remainder);
	}
}



// ----------------------------------------------------------------------------
// Precond::partBandedFwdElim()
// Precond::partBandedFwdElim_const()
// Precond::partBandedFwdElim_var()
//
// This function performs the forward elimination sweep for the given banded
// matrix B (assumed to encode the LU factors) and vector v.
// ----------------------------------------------------------------------------
template <typename Vector>
void 
Precond<Vector>::partBandedFwdSweep(Vector&  v)
{
	if (!m_variousBandwidth)
		partBandedFwdSweep_const(v);
	else
		partBandedFwdSweep_var(v);
}

template <typename Vector>
void 
Precond<Vector>::partBandedFwdSweep_const(Vector&  v)
{
	ValueType* p_B = thrust::raw_pointer_cast(&m_B[0]);
	ValueType* p_v = thrust::raw_pointer_cast(&v[0]);

	int partSize   = m_n / m_numPartitions;
	int remainder  = m_n % m_numPartitions;

	if (m_precondMethod == Block || m_method == LU_only || m_numPartitions == 1) {
		if (m_k > 1024)
			device::forwardElimL_general<ValueType, ValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else if (m_k > 32)
			device::forwardElimL_g32<ValueType, ValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else
			device::forwardElimL<ValueType, ValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
	} else {
		if (m_k > 1024)
			device::forwardElimL_LU_UL_general<ValueType, ValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else if (m_k > 32)
			device::forwardElimL_LU_UL_g32<ValueType, ValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else
			device::forwardElimL_LU_UL<ValueType, ValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
	}
}

template <typename Vector>
void 
Precond<Vector>::partBandedFwdSweep_var(Vector&  v)
{
	ValueType* p_B = thrust::raw_pointer_cast(&m_B[0]);
	ValueType* p_v = thrust::raw_pointer_cast(&v[0]);

	int* p_ks = thrust::raw_pointer_cast(&m_ks[0]);
	int tmp_k = cusp::blas::nrmmax(m_ks);
	int* p_BOffsets = thrust::raw_pointer_cast(&m_BOffsets[0]);

	int partSize   = m_n / m_numPartitions;
	int remainder  = m_n % m_numPartitions;

	if (tmp_k > 1024)
		device::var::fwdElim_sol<ValueType, ValueType><<<m_numPartitions, 512>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
	else if (tmp_k > 32)
		device::var::fwdElim_sol_medium<ValueType, ValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
	else
		device::var::fwdElim_sol_narrow<ValueType, ValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
}


// ----------------------------------------------------------------------------
// Precond::partBandedBckSweep()
// Precond::partBandedBckSweep_const()
// Precond::partBandedBckSweep_var()
//
// This function performs the backward substitution sweep for the given banded
// matrix B (assumed to encode the LU factors) and vector v.
// ----------------------------------------------------------------------------
template <typename Vector>
void 
Precond<Vector>::partBandedBckSweep(Vector&  v)
{
	if (!m_variousBandwidth)
		partBandedBckSweep_const(v);
	else
		partBandedBckSweep_var(v);
}

template <typename Vector>
void 
Precond<Vector>::partBandedBckSweep_const(Vector&  v)
{
	ValueType* p_B = thrust::raw_pointer_cast(&m_B[0]);
	ValueType* p_v = thrust::raw_pointer_cast(&v[0]);

	int partSize   = m_n / m_numPartitions;
	int remainder  = m_n % m_numPartitions;

	if (m_precondMethod == Block || m_method == LU_only || m_numPartitions == 1) {
		if (m_numPartitions > 1) {
			if (m_k > 1024)
				device::backwardElimU_general<ValueType, ValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
			else if (m_k > 32)
				device::backwardElimU_g32<ValueType, ValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
			else
				device::backwardElimU<ValueType, ValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		} else {
			int gridX = 1, blockX = m_n;
			if (blockX > BLOCK_SIZE) {
				gridX = (blockX + BLOCK_SIZE - 1) / BLOCK_SIZE;
				blockX = BLOCK_SIZE;
			}
			dim3 grids(gridX, m_numPartitions);
			device::preBck_sol_divide<ValueType, ValueType><<<grids, blockX>>>(m_n, m_k, p_B, p_v, partSize, remainder);

			if (m_k > 1024)
				device::bckElim_sol<ValueType, ValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
			else if (m_k > 32)
				device::bckElim_sol_medium<ValueType, ValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
			else
				device::bckElim_sol_narrow<ValueType, ValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		}
	} else {
		if (m_k > 1024)
			device::backwardElimU_LU_UL_general<ValueType, ValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else if (m_k > 32)
			device::backwardElimU_LU_UL_g32<ValueType, ValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else
			device::backwardElimU_LU_UL<ValueType, ValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
	}
}

template <typename Vector>
void 
Precond<Vector>::partBandedBckSweep_var(Vector&  v)
{
	ValueType* p_B = thrust::raw_pointer_cast(&m_B[0]);
	ValueType* p_v = thrust::raw_pointer_cast(&v[0]);

	int* p_ks = thrust::raw_pointer_cast(&m_ks[0]);
	int tmp_k = cusp::blas::nrmmax(m_ks);
	int* p_BOffsets = thrust::raw_pointer_cast(&m_BOffsets[0]);

	int partSize   = m_n / m_numPartitions;
	int remainder  = m_n % m_numPartitions;

	int gridX = 1, blockX = partSize + 1;
	kernelConfigAdjust(blockX, gridX, BLOCK_SIZE);
	dim3 grids(gridX, m_numPartitions);
	device::var::preBck_sol_divide<ValueType, ValueType><<<grids, blockX>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);

	if (tmp_k > 1024)
		device::var::bckElim_sol<ValueType, ValueType><<<m_numPartitions, 512>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
	else if (tmp_k > 32) 
		device::var::bckElim_sol_medium<ValueType, ValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
	else
		device::var::bckElim_sol_narrow<ValueType, ValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
}


// ----------------------------------------------------------------------------
// Precond::partFullFwdSweep()
//
// This function performs the forward elimination sweep for the given full
// matrix R (assumed to encode the LU factors) and vector v.
// ----------------------------------------------------------------------------
template <typename Vector>
void 
Precond<Vector>::partFullFwdSweep(Vector&  v)
{
	ValueType* p_R = thrust::raw_pointer_cast(&m_R[0]);
	ValueType* p_v = thrust::raw_pointer_cast(&v[0]);

	int partSize   = m_n / m_numPartitions;
	int remainder  = m_n % m_numPartitions;

	dim3 grids(m_numPartitions-1, 1);

	if (!m_variousBandwidth) {
		if (m_k > 512)
			device::forwardElimLNormal_g512<ValueType, ValueType><<<grids, 512>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
		else
			device::forwardElimLNormal<ValueType, ValueType><<<grids, 2*m_k-1>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
	} else {
		int* p_ROffsets = thrust::raw_pointer_cast(&m_ROffsets[0]);
		int* p_spike_ks = thrust::raw_pointer_cast(&m_spike_ks[0]);

		if (m_k > 512)
			device::var::fwdElim_full<ValueType, ValueType><<<grids, 512>>>(m_n, p_spike_ks,  p_ROffsets, p_R, p_v, partSize, remainder);
		else
			device::var::fwdElim_full_narrow<ValueType, ValueType><<<grids, m_k>>>(m_n, p_spike_ks, p_ROffsets, p_R, p_v, partSize, remainder);
	}
}



// ----------------------------------------------------------------------------
// Precond::partFullBckSweep()
//
// This function performs the backward substitution sweep for the given full
// matrix R (assumed to encode the LU factors) and vector v.
// ----------------------------------------------------------------------------
template <typename Vector>
void 
Precond<Vector>::partFullBckSweep(Vector&  v)
{
	ValueType* p_R = thrust::raw_pointer_cast(&m_R[0]);
	ValueType* p_v = thrust::raw_pointer_cast(&v[0]);

	int partSize   = m_n / m_numPartitions;
	int remainder  = m_n % m_numPartitions;

	dim3 grids(m_numPartitions-1, 1);

	if (!m_variousBandwidth) {
		if (m_k > 512)
			device::backwardElimUNormal_g512<ValueType, ValueType><<<grids, 512>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
		else
			device::backwardElimUNormal<ValueType, ValueType><<<grids, 2*m_k-1>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
	} else {
		int* p_ROffsets = thrust::raw_pointer_cast(&m_ROffsets[0]);
		int* p_spike_ks = thrust::raw_pointer_cast(&m_spike_ks[0]);

		if (m_k > 512) {
			device::var::preBck_full_divide<ValueType, ValueType><<<m_numPartitions-1, 512>>>(m_n, p_spike_ks, p_ROffsets, p_R, p_v, partSize, remainder);
			device::var::bckElim_full<ValueType, ValueType><<<grids, 512>>>(m_n, p_spike_ks, p_ROffsets, p_R, p_v, partSize, remainder);
		}
		else {
			device::var::preBck_full_divide_narrow<ValueType, ValueType><<<m_numPartitions-1, m_k>>>(m_n, p_spike_ks, p_ROffsets, p_R, p_v, partSize, remainder);
			device::var::bckElim_full_narrow<ValueType, ValueType><<<grids, 2*m_k-1>>>(m_n, p_spike_ks, p_ROffsets, p_R, p_v, partSize, remainder);
		}
	}
}


// ----------------------------------------------------------------------------
// Precond::purifyRHS()
//
// This function applies the purification step by performing a specialized
// inner product between the off-diagonal blocks of the original matrix
// and the vector 'v'. The result is stored in the output vector 'res'.
// ----------------------------------------------------------------------------
template <typename Vector>
void 
Precond<Vector>::purifyRHS(Vector&  v,
                           Vector&  res)
{
	ValueType* p_offDiags = thrust::raw_pointer_cast(&m_offDiags[0]);
	ValueType* p_v        = thrust::raw_pointer_cast(&v[0]);
	ValueType* p_res      = thrust::raw_pointer_cast(&res[0]);

	int partSize   = m_n / m_numPartitions;
	int remainder  = m_n % m_numPartitions;

	dim3 grids(m_k, m_numPartitions-1);

	if (!m_variousBandwidth) {
		if (m_k > 256)
			device::innerProductBCX_g256<ValueType, ValueType><<<grids, 256>>>(p_offDiags, p_v, p_res, m_n, m_k, partSize, m_numPartitions, remainder);
		else if (m_k > 64)
			device::innerProductBCX_g64<ValueType, ValueType><<<grids, 256>>>(p_offDiags, p_v, p_res, m_n, m_k, partSize, m_numPartitions, remainder);
		else if (m_k > 32)
			device::innerProductBCX_g32<ValueType, ValueType><<<grids, 64>>>(p_offDiags, p_v, p_res, m_n, m_k, partSize, m_numPartitions, remainder);
		else
			device::innerProductBCX<ValueType, ValueType><<<grids, 32>>>(p_offDiags, p_v, p_res, m_n, m_k, partSize, m_numPartitions, remainder);
	} else {
		int* p_WVOffsets = thrust::raw_pointer_cast(&m_WVOffsets[0]);
		int* p_spike_ks = thrust::raw_pointer_cast(&m_spike_ks[0]);
		
		if (m_k > 256)
			device::innerProductBCX_var_bandwidth_g256<ValueType, ValueType><<<grids, 256>>>(p_offDiags, p_v, p_res, m_n, p_spike_ks, p_WVOffsets, partSize, m_numPartitions, remainder);
		else if (m_k > 64)
			device::innerProductBCX_var_bandwidth_g64<ValueType, ValueType><<<grids, 256>>>(p_offDiags, p_v, p_res, m_n, p_spike_ks, p_WVOffsets, partSize, m_numPartitions, remainder);
		else if (m_k > 32)
			device::innerProductBCX_var_bandwidth_g32<ValueType, ValueType><<<grids, 64>>>(p_offDiags, p_v, p_res, m_n, p_spike_ks, p_WVOffsets, partSize, m_numPartitions, remainder);
		else
			device::innerProductBCX_var_bandwidth<ValueType, ValueType><<<grids, 32>>>(p_offDiags, p_v, p_res, m_n, p_spike_ks, p_WVOffsets, partSize, m_numPartitions, remainder);
	}
}


// ----------------------------------------------------------------------------
// Precond::calculateSpikes()
// Precond::calculateSpikes_const()
// Precond::calculateSpikes_var()
//
// This function calculates the spike blocks in the LU_only case.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::calculateSpikes(Vector&  WV)
{
	if (!m_variousBandwidth)
		calculateSpikes_const(WV);
	else {
		int totalRHSCount = cusp::blas::nrm1(m_offDiagWidths_right_host) + cusp::blas::nrm1(m_offDiagWidths_left_host);
		if (totalRHSCount >= 2800)
			calculateSpikes_var(WV);
		else
			calculateSpikes_var_old(WV);
	}
}

template <typename Vector>
void
Precond<Vector>::calculateSpikes_var_old(Vector&  WV)
{
	ValueType* p_WV = thrust::raw_pointer_cast(&WV[0]);

	Vector WV_spare(m_k*m_k);
	ValueType* p_WV_spare = thrust::raw_pointer_cast(&WV_spare[0]);

	// Calculate the size of the first and last partitions.
	int last_partition_size = m_n / m_numPartitions;
	int first_partition_size = last_partition_size;
	int numThreadsToUse = adjustNumThreads(cusp::blas::nrm1(m_ks) / m_numPartitions);

	if (m_n % m_numPartitions != 0)
		first_partition_size++;


	// Copy WV into extV, perform sweeps to calculate extV, then copy back extV to WV.
	// Note that we skip the last partition (no right spike associated with it). Also
	// note that we only perform truncated spikes using the bottom parts of the L and
	// U factors to calculate the bottom block of the right spikes V.
	{
		int  n_eff       = m_n - last_partition_size;
		int  numPart_eff = m_numPartitions - 1;
		int  partSize    = n_eff / numPart_eff;
		int  remainder   = n_eff % numPart_eff;

		const int BUF_FACTOR = 16;

		Vector extV(m_k * n_eff, 0), buffer;

		ValueType* p_extV = thrust::raw_pointer_cast(&extV[0]);
		ValueType* p_B    = thrust::raw_pointer_cast(&m_B[0]);
		int *p_secondReordering = thrust::raw_pointer_cast(&m_secondReordering[0]);
		int *p_secondPerm = thrust::raw_pointer_cast(&m_secondPerm[0]);

		dim3 gridsCopy(m_k, numPart_eff);
		dim3 gridsSweep(numPart_eff, m_k);

		int *p_ks = thrust::raw_pointer_cast(&m_ks[0]);
		int *p_offDiagWidths_right = thrust::raw_pointer_cast(&m_offDiagWidths_right[0]);
		int *p_offDiagPerms_right = thrust::raw_pointer_cast(&m_offDiagPerms_right[0]);
		int *p_first_rows = thrust::raw_pointer_cast(&m_first_rows[0]);
		int *p_offsets = thrust::raw_pointer_cast(&m_BOffsets[0]);

		int permuteBlockX = n_eff, permuteGridX = 1;
		kernelConfigAdjust(permuteBlockX, permuteGridX, BLOCK_SIZE);
		dim3 gridsPermute(permuteGridX, m_k);

		{
			device::copyWVFromOrToExtendedV_general<ValueType><<<gridsCopy, numThreadsToUse>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, false);
			buffer.resize((m_k - (BUF_FACTOR-1) * (m_k / BUF_FACTOR)) * n_eff);
			ValueType* p_buffer = thrust::raw_pointer_cast(&buffer[0]);

			for (int i=0; i<BUF_FACTOR; i++) {
				gridsPermute.y = m_k / BUF_FACTOR;
				if (i == BUF_FACTOR - 1)
					gridsPermute.y = m_k - (BUF_FACTOR-1) * (m_k/BUF_FACTOR);
				device::permute<ValueType><<<gridsPermute, permuteBlockX>>>(n_eff, p_extV+(i*(m_k/BUF_FACTOR)*n_eff), p_buffer, p_secondPerm);
				thrust::copy(buffer.begin(), buffer.begin()+(gridsPermute.y * n_eff), extV.begin()+i*(m_k/BUF_FACTOR)*n_eff);
			}

			{
				int last_row = 0, pseudo_first_row = 0;
				for (int i=0; i<numPart_eff; i++) {
					if (i < remainder)
						last_row += partSize + 1;
					else 
						last_row += partSize;

					int tmp_first_row = m_first_rows_host[i];
					device::var::fwdElim_rightSpike_per_partition<ValueType, ValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>> (n_eff, m_ks_host[i], m_BOffsets_host[i]+m_ks_host[i]+(2*m_ks_host[i]+1)*(m_first_rows_host[i]-pseudo_first_row), p_B, p_extV, m_first_rows_host[i], last_row);
					
					int blockX = last_row - m_first_rows_host[i], gridX = 1;
					kernelConfigAdjust(blockX, gridX, BLOCK_SIZE);
					dim3 grids(gridX, m_offDiagWidths_right_host[i]);
					device::var::preBck_rightSpike_divide_per_partition<ValueType, ValueType><<<grids, blockX>>> (n_eff, m_ks_host[i], m_BOffsets_host[i]+m_ks_host[i]+(2*m_ks_host[i]+1)*(m_first_rows_host[i]-pseudo_first_row), p_B, p_extV, m_first_rows_host[i], last_row);

					m_first_rows_host[i] = thrust::reduce(m_secondPerm.begin()+(last_row-m_k), m_secondPerm.begin()+last_row, last_row, thrust::minimum<int>());
					device::var::bckElim_rightSpike_per_partition<ValueType, ValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>> (n_eff, m_ks_host[i], m_BOffsets_host[i]+m_ks_host[i]+(2*m_ks_host[i]+1)*(last_row-pseudo_first_row-1), p_B, p_extV, m_first_rows_host[i], last_row);

					pseudo_first_row = last_row;
				}
			}

			for (int i=0; i<BUF_FACTOR; i++) {
				gridsPermute.y = m_k / BUF_FACTOR;
				if (i == BUF_FACTOR - 1)
					gridsPermute.y = m_k - (BUF_FACTOR-1) * (m_k/BUF_FACTOR);
				device::permute<ValueType><<<gridsPermute, permuteBlockX>>>(n_eff, p_extV+(i*(m_k/BUF_FACTOR)*n_eff), p_buffer, p_secondReordering);
				thrust::copy(buffer.begin(), buffer.begin()+(gridsPermute.y * n_eff), extV.begin()+i*(m_k/BUF_FACTOR)*n_eff);
			}

			device::copyWVFromOrToExtendedV_general<ValueType><<<gridsCopy, numThreadsToUse>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, true);
		}
		for (int i=0; i<numPart_eff; i++) {
			cusp::blas::fill(WV_spare, 0);
			device::matrixVReordering_perPartition<ValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>>(m_k, p_WV+2*i*m_k*m_k, p_WV_spare, p_offDiagPerms_right+i*m_k);
			thrust::copy(WV_spare.begin(), WV_spare.end(), WV.begin() + (2*i*m_k*m_k));
		}
	}


	// Copy WV into extW, perform sweeps to calculate extW, then copy back extW to WV.
	// Note that we skip the first partition (no left spike associated with it). Also
	// note that we perform full sweeps using the L and U factors to calculate the
	// entire left spikes W.
	{
		int  n_eff       = m_n - first_partition_size;
		int  numPart_eff = m_numPartitions - 1;
		int  partSize    = n_eff / numPart_eff;
		int  remainder   = n_eff % numPart_eff;

		const int BUF_FACTOR = 16;

		Vector  extW(m_k * n_eff, 0), buffer;

		ValueType* p_extW = thrust::raw_pointer_cast(&extW[0]);
		ValueType* p_B = thrust::raw_pointer_cast(&m_B[0]);

		dim3 gridsSweep(numPart_eff, m_k);
		dim3 gridsCopy(m_k, numPart_eff);

		int *p_ks = thrust::raw_pointer_cast(&m_ks[1]);
		int *p_offDiagWidths_left = thrust::raw_pointer_cast(&m_offDiagWidths_left[0]);
		int *p_offDiagPerms_left = thrust::raw_pointer_cast(&m_offDiagPerms_left[0]);
		VectorI tmp_offsets;
		VectorI tmp_secondReordering(m_n, first_partition_size);
		VectorI tmp_secondPerm(m_n, first_partition_size);

		cusp::blas::axpby(m_secondReordering, tmp_secondReordering, tmp_secondReordering, 1.0, -1.0);
		cusp::blas::axpby(m_secondPerm, tmp_secondPerm, tmp_secondPerm, 1.0, -1.0);

		int *p_secondReordering = thrust::raw_pointer_cast(&tmp_secondReordering[first_partition_size]);
		int *p_secondPerm = thrust::raw_pointer_cast(&tmp_secondPerm[first_partition_size]);
		{
			cusp::array1d<int, cusp::host_memory> tmp_offsets_host = m_BOffsets;
			for (int i=m_numPartitions-1; i>=1; i--)
				tmp_offsets_host[i] -= tmp_offsets_host[1];
			tmp_offsets = tmp_offsets_host;
		}
		int *p_offsets = thrust::raw_pointer_cast(&tmp_offsets[1]);

		int permuteBlockX = n_eff, permuteGridX = 1;
		kernelConfigAdjust(permuteBlockX, permuteGridX, BLOCK_SIZE);
		dim3 gridsPermute(permuteGridX, m_k);

		{
			device::copyWVFromOrToExtendedW_general<ValueType><<<gridsCopy, numThreadsToUse>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, false);
			buffer.resize((m_k - (BUF_FACTOR-1) * (m_k / BUF_FACTOR)) * n_eff);
			ValueType* p_buffer = thrust::raw_pointer_cast(&buffer[0]);

			for (int i=0; i<BUF_FACTOR; i++) {
				gridsPermute.y = m_k / BUF_FACTOR;
				if (i == BUF_FACTOR - 1)
					gridsPermute.y = m_k - (BUF_FACTOR-1) * (m_k/BUF_FACTOR);
				device::permute<ValueType><<<gridsPermute, permuteBlockX>>>(n_eff, p_extW+i*(m_k/BUF_FACTOR)*n_eff, p_buffer, p_secondPerm);
				thrust::copy(buffer.begin(), buffer.begin()+(gridsPermute.y * n_eff), extW.begin()+i*(m_k/BUF_FACTOR)*n_eff);
			}

			{
				int last_row = 0, first_row = 0;
				for (int i=0; i<numPart_eff; i++) {
					if (i < remainder)
						last_row += partSize + 1;
					else 
						last_row += partSize;
					device::var::fwdElim_leftSpike_per_partition<ValueType, ValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>> (n_eff, m_ks_host[i+1], m_k - m_offDiagWidths_left_host[i], m_BOffsets_host[i+1]+m_ks_host[i+1], p_B, p_extW, first_row, last_row);
					int blockX = last_row - first_row, gridX = 1;
					kernelConfigAdjust(blockX, gridX, BLOCK_SIZE);
					dim3 grids(gridX, m_offDiagWidths_left_host[i]);
					device::var::preBck_leftSpike_divide_per_partition<ValueType, ValueType><<<grids, blockX>>> (n_eff, m_ks_host[i+1], m_k - m_offDiagWidths_left_host[i], m_BOffsets_host[i+1]+m_ks_host[i+1], p_B, p_extW, first_row, last_row);
					device::var::bckElim_leftSpike_per_partition<ValueType, ValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>>(n_eff, m_ks_host[i+1], m_k - m_offDiagWidths_left_host[i], m_BOffsets_host[i+1] + m_ks_host[i+1] + (2*m_ks_host[i+1]+1)*(last_row-first_row-1), p_B, p_extW, first_row, last_row);
					first_row = last_row;
				}
			}

			for (int i=0; i<BUF_FACTOR; i++) {
				gridsPermute.y = m_k / BUF_FACTOR;
				if (i == BUF_FACTOR - 1)
					gridsPermute.y = m_k - (BUF_FACTOR-1) * (m_k/BUF_FACTOR);

				device::permute<ValueType><<<gridsPermute, permuteBlockX>>>(n_eff, p_extW+i*(m_k/BUF_FACTOR)*n_eff, p_buffer, p_secondReordering);
				thrust::copy(buffer.begin(), buffer.begin()+(gridsPermute.y * n_eff), extW.begin()+i*(m_k/BUF_FACTOR)*n_eff);
			}

			device::copyWVFromOrToExtendedW_general<ValueType><<<gridsCopy, numThreadsToUse>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, true);
		}

		for (int i=0; i<numPart_eff; i++) {
			cusp::blas::fill(WV_spare, 0);
			device::matrixWReordering_perPartition<ValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>>(m_k, p_WV+(2*i+1)*m_k*m_k, p_WV_spare, p_offDiagPerms_left+i*m_k);
			thrust::copy(WV_spare.begin(), WV_spare.end(), WV.begin() + ((2*i+1)*m_k*m_k));
		}
	}
}
template <typename Vector>
void
Precond<Vector>::calculateSpikes_const(Vector&  WV)
{
	ValueType* p_WV = thrust::raw_pointer_cast(&WV[0]);


	// Calculate the size of the first and last partitions.
	int last_partition_size = m_n / m_numPartitions;
	int first_partition_size = last_partition_size;

	if (m_n % m_numPartitions != 0)
		first_partition_size++;


	// Copy WV into extV, perform sweeps to calculate extV, then copy back extV to WV.
	// Note that we skip the last partition (no right spike associated with it). Also
	// note that we only perform truncated spikes using the bottom parts of the L and
	// U factors to calculate the bottom block of the right spikes V.
	{
		int  n_eff       = m_n - last_partition_size;
		int  numPart_eff = m_numPartitions - 1;
		int  partSize    = n_eff / numPart_eff;
		int  remainder   = n_eff % numPart_eff;

		Vector extV(m_k * n_eff, 0);

		ValueType* p_extV = thrust::raw_pointer_cast(&extV[0]);
		ValueType* p_B    = thrust::raw_pointer_cast(&m_B[0]);

		dim3 gridsCopy(m_k, numPart_eff);
		dim3 gridsSweep(numPart_eff, m_k);

		if (m_k > 1024) {
			device::copyWVFromOrToExtendedV_general<ValueType><<<gridsCopy, 512>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, false);
			device::forwardElimL_bottom_general<ValueType, ValueType><<<gridsSweep, 512>>>(n_eff, m_k, m_k, p_B, p_extV, partSize, remainder);
			device::backwardElimU_bottom_general<ValueType, ValueType><<<gridsSweep, 512>>>(n_eff, m_k, 2*m_k, p_B, p_extV, partSize, remainder);
			device::copyWVFromOrToExtendedV_general<ValueType><<<gridsCopy, 512>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, true);
		} else if (m_k > 32) {
			device::copyWVFromOrToExtendedV<ValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, false);
			device::forwardElimL_bottom_g32<ValueType, ValueType><<<gridsSweep, m_k>>>(n_eff, m_k, m_k, p_B, p_extV, partSize, remainder);
			device::backwardElimU_bottom_g32<ValueType, ValueType><<<gridsSweep, m_k>>>(n_eff, m_k, 2*m_k, p_B, p_extV, partSize, remainder);
			device::copyWVFromOrToExtendedV<ValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, true);
		} else {
			device::copyWVFromOrToExtendedV<ValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, false);
			device::forwardElimL_bottom<ValueType, ValueType><<<gridsSweep, m_k>>>(n_eff, m_k, m_k, p_B, p_extV, partSize, remainder);
			device::backwardElimU_bottom<ValueType, ValueType><<<gridsSweep, m_k>>>(n_eff, m_k, 2*m_k, p_B, p_extV, partSize, remainder);
			device::copyWVFromOrToExtendedV<ValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, true);
		}
	}


	// Copy WV into extW, perform sweeps to calculate extW, then copy back extW to WV.
	// Note that we skip the first partition (no left spike associated with it). Also
	// note that we perform full sweeps using the L and U factors to calculate the
	// entire left spikes W.
	{
		int  n_eff       = m_n - first_partition_size;
		int  numPart_eff = m_numPartitions - 1;
		int  partSize    = n_eff / numPart_eff;
		int  remainder   = n_eff % numPart_eff;

		Vector  extW(m_k * n_eff, 0);

		ValueType* p_extW = thrust::raw_pointer_cast(&extW[0]);
		ValueType* p_B    = thrust::raw_pointer_cast(&m_B[(2*m_k+1)*first_partition_size]);

		dim3 gridsSweep(numPart_eff, m_k);
		dim3 gridsCopy(m_k, numPart_eff);

		if (m_k > 1024) {
			device::copyWVFromOrToExtendedW_general<ValueType><<<gridsCopy, 512>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, false);
			device::forwardElimL_general<ValueType, ValueType><<<gridsSweep, 512>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::backwardElimU_general<ValueType, ValueType><<<gridsSweep, 512>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::copyWVFromOrToExtendedW_general<ValueType><<<gridsCopy, 512>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, true);
		} else if (m_k > 32) {
			device::copyWVFromOrToExtendedW<ValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, false);
			device::forwardElimL_g32<ValueType, ValueType><<<gridsSweep, m_k>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::backwardElimU_g32<ValueType, ValueType><<<gridsSweep, m_k>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::copyWVFromOrToExtendedW<ValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, true);
		} else {
			device::copyWVFromOrToExtendedW<ValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, false);
			device::forwardElimL<ValueType, ValueType><<<gridsSweep, m_k>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::backwardElimU<ValueType, ValueType><<<gridsSweep, m_k>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::copyWVFromOrToExtendedW<ValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, true);
		}
	}
}

template <typename Vector>
void
Precond<Vector>::calculateSpikes_var(Vector&  WV)
{
	ValueType* p_WV = thrust::raw_pointer_cast(&WV[0]);

	Vector WV_spare(m_k*m_k);
	ValueType* p_WV_spare = thrust::raw_pointer_cast(&WV_spare[0]);

	// Calculate the size of the first and last partitions.
	int numThreadsToUse = adjustNumThreads(cusp::blas::nrm1(m_ks) / m_numPartitions);

	const int SWEEP_MAX_NUM_THREADS = 128;
	// Copy WV into extV, perform sweeps to calculate extV, then copy back extV to WV.
	// Note that we skip the last partition (no right spike associated with it). Also
	// note that we only perform truncated spikes using the bottom parts of the L and
	// U factors to calculate the bottom block of the right spikes V.
	{
		int  n_eff       = m_n;
		int  numPart_eff = m_numPartitions;
		int  partSize    = n_eff / numPart_eff;
		int  remainder   = n_eff % numPart_eff;
		int rightOffDiagWidth = cusp::blas::nrmmax(m_offDiagWidths_right);
		int leftOffDiagWidth = cusp::blas::nrmmax(m_offDiagWidths_left);

		Vector extWV((leftOffDiagWidth + rightOffDiagWidth) * n_eff, 0), buffer;

		ValueType* p_extWV = thrust::raw_pointer_cast(&extWV[0]);
		ValueType* p_B    = thrust::raw_pointer_cast(&m_B[0]);
		int *p_secondReordering = thrust::raw_pointer_cast(&m_secondReordering[0]);
		int *p_secondPerm = thrust::raw_pointer_cast(&m_secondPerm[0]);

		int *p_ks = thrust::raw_pointer_cast(&m_ks[0]);
		int *p_offDiagWidths_right = thrust::raw_pointer_cast(&m_offDiagWidths_right[0]);
		int *p_offDiagPerms_right = thrust::raw_pointer_cast(&m_offDiagPerms_right[0]);
		int *p_offDiagWidths_left = thrust::raw_pointer_cast(&m_offDiagWidths_left[0]);
		int *p_offDiagPerms_left = thrust::raw_pointer_cast(&m_offDiagPerms_left[0]);
		int *p_first_rows = thrust::raw_pointer_cast(&m_first_rows[0]);
		int *p_offsets = thrust::raw_pointer_cast(&m_BOffsets[0]);

		int permuteBlockX = leftOffDiagWidth+rightOffDiagWidth, permuteGridX = 1, permuteGridY = n_eff, permuteGridZ = 1;
		kernelConfigAdjust(permuteBlockX, permuteGridX, BLOCK_SIZE);
		kernelConfigAdjust(permuteGridY, permuteGridZ, MAX_GRID_DIMENSION);
		dim3 gridsPermute(permuteGridX, permuteGridY, permuteGridZ);

		buffer.resize((leftOffDiagWidth + rightOffDiagWidth) * n_eff);
		ValueType *p_buffer = thrust::raw_pointer_cast(&buffer[0]);

		dim3 gridsCopy((leftOffDiagWidth + rightOffDiagWidth), numPart_eff);

		device::copyWVFromOrToExtendedWVTranspose_general<ValueType><<<gridsCopy, numThreadsToUse>>>(leftOffDiagWidth + rightOffDiagWidth, m_k, rightOffDiagWidth, partSize, remainder, m_k-rightOffDiagWidth-leftOffDiagWidth, p_WV, p_extWV, false);
		device::columnPermute<ValueType><<<gridsPermute, permuteBlockX>>>(n_eff, leftOffDiagWidth+rightOffDiagWidth, p_extWV, p_buffer, p_secondPerm);

		{				
			int sweepBlockX = leftOffDiagWidth, sweepGridX = 1;
			if (sweepBlockX < rightOffDiagWidth)
				sweepBlockX = rightOffDiagWidth;
			kernelConfigAdjust(sweepBlockX, sweepGridX, SWEEP_MAX_NUM_THREADS);
			dim3 sweepGrids(sweepGridX, 2*numPart_eff-2);

			device::var::fwdElim_spike<ValueType, ValueType><<<sweepGrids, sweepBlockX>>>(n_eff, p_ks, leftOffDiagWidth + rightOffDiagWidth, rightOffDiagWidth, p_offsets, p_B, p_buffer, partSize, remainder, p_offDiagWidths_left, p_offDiagWidths_right, p_first_rows);

			int preBckBlockX = leftOffDiagWidth + rightOffDiagWidth, preBckGridX = 1, preBckGridY = n_eff, preBckGridZ = 1;
			kernelConfigAdjust(preBckBlockX, preBckGridX, BLOCK_SIZE);
			kernelConfigAdjust(preBckGridY, preBckGridZ, MAX_GRID_DIMENSION);
			dim3 preBckGrids(preBckGridX, preBckGridY, preBckGridZ);

			device::var::preBck_offDiag_divide<ValueType, ValueType><<<preBckGrids, preBckBlockX>>>(n_eff, leftOffDiagWidth + rightOffDiagWidth, p_ks, p_offsets, p_B, p_buffer, partSize, remainder);

			{
				int last_row = 0;
				for (int i=0; i<m_numPartitions - 1; i++) {
					if (i < remainder)
						last_row += (partSize + 1);
					else
						last_row += partSize;

					m_first_rows_host[i] = thrust::reduce(m_secondPerm.begin()+(last_row-m_k), m_secondPerm.begin()+last_row, last_row, thrust::minimum<int>());
				}
				m_first_rows = m_first_rows_host;
				p_first_rows = thrust::raw_pointer_cast(&m_first_rows[0]);
			}

			device::var::bckElim_spike<ValueType, ValueType><<<sweepGrids, sweepBlockX>>>(n_eff, p_ks, leftOffDiagWidth + rightOffDiagWidth, rightOffDiagWidth, p_offsets, p_B, p_buffer, partSize, remainder, p_offDiagWidths_left, p_offDiagWidths_right, p_first_rows);
		}


		device::columnPermute<ValueType><<<gridsPermute, permuteBlockX>>>(n_eff, leftOffDiagWidth + rightOffDiagWidth, p_buffer, p_extWV, p_secondReordering);
		device::copyWVFromOrToExtendedWVTranspose_general<ValueType><<<gridsCopy, numThreadsToUse>>>(leftOffDiagWidth + rightOffDiagWidth, m_k, rightOffDiagWidth, partSize, remainder, m_k-rightOffDiagWidth-leftOffDiagWidth, p_WV, p_extWV, true);

		for (int i=0; i<numPart_eff-1; i++) {
			cusp::blas::fill(WV_spare, 0);
			device::matrixVReordering_perPartition<ValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>>(m_k, p_WV+2*i*m_k*m_k, p_WV_spare, p_offDiagPerms_right+i*m_k);
			thrust::copy(WV_spare.begin(), WV_spare.end(), WV.begin()+(2*i*m_k*m_k));

			cusp::blas::fill(WV_spare, 0);
			device::matrixWReordering_perPartition<ValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>>(m_k, p_WV+(2*i+1)*m_k*m_k, p_WV_spare, p_offDiagPerms_left+i*m_k);
			thrust::copy(WV_spare.begin(), WV_spare.end(), WV.begin()+((2*i+1)*m_k*m_k));
		}
	}
}


// ----------------------------------------------------------------------------
// Precond::calculateSpikes
//
// This function adjust the number of threads used for kernels which can take
// any number of threads.
// ----------------------------------------------------------------------------
template <typename Vector>
int
Precond<Vector>::adjustNumThreads(int inNumThreads) {
	int prev = 0, cur;
	for (int i=0; i<16; i++) {
		cur = (i+1) << 5;
		if (inNumThreads > cur) {
			prev = cur;
			continue;
		}
		if (inNumThreads - prev > cur - inNumThreads || prev == 0)
			return cur;
		return prev;
	}
	return 512;
}


// ----------------------------------------------------------------------------
// Precond::calculateSpikes
//
// This function calculates the spike blocks in the LU_UL case.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::calculateSpikes(Vector&  B2,
                                 Vector&  WV)
{
	int  two_k     = 2 * m_k;
	int  partSize  = m_n / m_numPartitions;
	int  remainder = m_n % m_numPartitions;

	// Compress the provided UL factorization 'B2' into 'compB2'.
	Vector compB2((two_k+1)*two_k*(m_numPartitions-1));
	cusp::blas::fill(compB2, 0);

	ValueType* p_B2 = thrust::raw_pointer_cast(&B2[0]);
	ValueType* p_compB2 = thrust::raw_pointer_cast(&compB2[0]);

	dim3 gridsCompress(two_k, m_numPartitions-1);

		if (m_k > 511)
			device::copydAtodA2_general<ValueType><<<gridsCompress, 1024>>>(m_n, m_k, p_B2, p_compB2, two_k, partSize, m_numPartitions, remainder);
		else
			device::copydAtodA2<ValueType><<<gridsCompress, two_k+1>>>(m_n, m_k, p_B2, p_compB2, two_k, partSize, m_numPartitions, remainder);

	// Combine 'B' and 'compB2' into 'partialB'.
	Vector partialB(2*(two_k+1)*(m_k+1)*(m_numPartitions-1));

	ValueType* p_B        = thrust::raw_pointer_cast(&m_B[0]);
	ValueType* p_partialB = thrust::raw_pointer_cast(&partialB[0]);

	dim3 gridsCopy(m_k+1, 2*(m_numPartitions-1));

	if (m_k > 511)
		device::copydAtoPartialA_general<ValueType><<<gridsCopy, 1024>>>(m_n, m_k, p_B, p_compB2, p_partialB, partSize, m_numPartitions, remainder, two_k);
	else
		device::copydAtoPartialA<ValueType><<<gridsCopy, two_k+1>>>(m_n, m_k, p_B, p_compB2, p_partialB, partSize, m_numPartitions, remainder, two_k);

	// Perform forward/backward sweeps to calculate the spike blocks 'W' and 'V'.
	ValueType* p_WV = thrust::raw_pointer_cast(&WV[0]);

	dim3 gridsSweep(m_numPartitions-1, m_k);

	if (m_k > 1024) {
		device::forwardElimLdWV_general<ValueType,ValueType><<<gridsSweep, 512>>>(m_k, p_partialB, p_WV, m_k, 0, 0);
		device::backwardElimUdWV_general<ValueType,ValueType><<<gridsSweep, 512>>>(m_k, p_partialB, p_WV, m_k, 0, 1);
		device::backwardElimUdWV_general<ValueType,ValueType><<<gridsSweep, 512>>>(m_k, p_partialB, p_WV, m_k, 1, 0);
		device::forwardElimLdWV_general<ValueType,ValueType><<<gridsSweep, 512>>>(m_k, p_partialB, p_WV, m_k, 1, 1);
	} else if (m_k > 32)  {
		device::forwardElimLdWV_g32<ValueType,ValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 0, 0);
		device::backwardElimUdWV_g32<ValueType,ValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 0, 1);
		device::backwardElimUdWV_g32<ValueType,ValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 1, 0);
		device::forwardElimLdWV_g32<ValueType,ValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 1, 1);
	} else {
		device::forwardElimLdWV<ValueType,ValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 0, 0);
		device::backwardElimUdWV<ValueType,ValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 0, 1);
		device::backwardElimUdWV<ValueType,ValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 1, 0);
		device::forwardElimLdWV<ValueType,ValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 1, 1);
	}
}


// ----------------------------------------------------------------------------
// assembleReducedMat()
//
// This function assembles the truncated Spike reduced matrix R.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::assembleReducedMat(Vector&  WV)
{
	ValueType* p_WV = thrust::raw_pointer_cast(&WV[0]);
	ValueType* p_R  = thrust::raw_pointer_cast(&m_R[0]);

	dim3 grids(m_k, m_numPartitions-1);

	if (!m_variousBandwidth) {
		if (m_k > 1024)
			device::assembleReducedMat_general<ValueType><<<grids, 512>>>(m_k, p_WV, p_R);
		else if (m_k > 32)
			device::assembleReducedMat_g32<ValueType><<<grids, m_k>>>(m_k, p_WV, p_R);
		else
			device::assembleReducedMat<ValueType><<<m_numPartitions-1, m_k*m_k>>>(m_k, p_WV, p_R);
	} else {
		int* p_WVOffsets = thrust::raw_pointer_cast(&m_WVOffsets[0]);
		int* p_ROffsets = thrust::raw_pointer_cast(&m_ROffsets[0]);
		int* p_spike_ks = thrust::raw_pointer_cast(&m_spike_ks[0]);
	
		if (m_k > 1024)
			device::assembleReducedMat_var_bandwidth_general<ValueType><<<grids, 512>>>(p_spike_ks, p_WVOffsets, p_ROffsets, p_WV, p_R);
		else if (m_k > 32)
			device::assembleReducedMat_var_bandwidth_g32<ValueType><<<grids, m_k>>>(p_spike_ks, p_WVOffsets, p_ROffsets, p_WV, p_R);
		else
			device::assembleReducedMat_var_bandwidth<ValueType><<<m_numPartitions-1, m_k*m_k>>>(p_spike_ks, p_WVOffsets, p_ROffsets, p_WV, p_R);
	}
}

// ----------------------------------------------------------------------------
// copyLastPartition()
//
// This function copies the last partition from B2, which contains the UL results,
// to m_B.
// ----------------------------------------------------------------------------
template <typename Vector>
void
Precond<Vector>::copyLastPartition(Vector &B2) {
	thrust::copy(B2.begin()+(2*m_k+1) * (m_n - m_n / m_numPartitions), B2.end(), m_B.begin()+(2*m_k+1) * (m_n - m_n / m_numPartitions) );
}



} // namespace spike


#endif
