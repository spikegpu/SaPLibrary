#ifndef SPIKE_PRECOND_CUH
#define SPIKE_PRECOND_CUH

#include <cusp/blas.h>
#include <cusp/print.h>

#include <thrust/logical.h>
#include <thrust/functional.h>

#include <spike/common.h>
#include <spike/graph.h>
#include <spike/timer.h>
#include <spike/strided_range.h>
#include <spike/device/factor_band_const.cuh>
#include <spike/device/factor_band_var.cuh>
#include <spike/device/sweep_band_const.cuh>
#include <spike/device/sweep_band_var.cuh>
#include <spike/device/inner_product.cuh>
#include <spike/device/shuffle.cuh>
#include <spike/device/data_transfer.cuh>

#include <omp.h>


namespace spike {

/**
 * This class encapsulates the truncated Spike preconditioner.
 */
template <typename PrecVector>
class Precond
{
public:
	typedef typename PrecVector::memory_space  MemorySpace;
	typedef typename PrecVector::value_type    PrecValueType;
	typedef typename PrecVector::iterator      PrecVectorIterator;

	typedef typename cusp::array1d<int, MemorySpace>                  IntVector;
	typedef IntVector                                                 MatrixMap;
	typedef typename cusp::array1d<PrecValueType, MemorySpace>        MatrixMapF;

	typedef typename cusp::array1d<PrecValueType, cusp::host_memory>  PrecVectorH;
	typedef typename cusp::array1d<int, cusp::host_memory>            IntVectorH;
	typedef IntVectorH                                                MatrixMapH;
	typedef typename cusp::array1d<PrecValueType, cusp::host_memory>  MatrixMapFH;

	typedef typename cusp::coo_matrix<int, PrecValueType, MemorySpace>        PrecMatrixCoo;
	typedef typename cusp::coo_matrix<int, PrecValueType, cusp::host_memory>  PrecMatrixCooH;


	Precond();

	Precond(int                 numPart,
	        bool                reorder,
	        bool                doMC64,
	        bool                scale,
	        double              dropOff_frac,
	        int                 maxBandwidth,
	        FactorizationMethod factMethod,
	        PreconditionerType  precondType,
	        bool                safeFactorization,
	        bool                variableBandwidth,
	        bool                trackReordering);

	Precond(const Precond&  prec);

	~Precond() {}

	Precond & operator = (const Precond &prec);

	double getTimeReorder() const         {return m_time_reorder;}
	double getTimeCPUAssemble() const     {return m_time_cpu_assemble;}
	double getTimeTransfer() const        {return m_time_transfer;}
	double getTimeToBanded() const        {return m_time_toBanded;}
	double getTimeCopyOffDiags() const    {return m_time_offDiags;}
	double getTimeBandLU() const          {return m_time_bandLU;}
	double getTimeBandUL() const          {return m_time_bandUL;}
	double gettimeAssembly() const        {return m_time_assembly;}
	double getTimeFullLU() const          {return m_time_fullLU;}
	double getTimeShuffle() const         {return m_time_shuffle;}

	int    getBandwidthReordering() const {return m_k_reorder;}
	int    getBandwidthMC64() const       {return m_k_mc64;}
	int    getBandwidth() const           {return m_k;}

	int    getNumPartitions() const       {return m_numPartitions;}
	double getActualDropOff() const       {return (double) m_dropOff_actual;}

	//// NOTE:  Matrix here will usually be PrecMatrixCooH, except
	////        when there's a single component when it will be whatever
	////        the user passes to Solver::setup().
	template <typename Matrix>
	void   setup(const Matrix&  A);

	bool   setupDone() const              {return m_setupDone;}

	void   update(const PrecVector& entries);

	void   solve(PrecVector& v, PrecVector& z);

private:
	int                  m_numPartitions;
	int                  m_n;
	int                  m_k;

	bool                 m_reorder;
	bool                 m_doMC64;
	bool                 m_scale;
	PrecValueType        m_dropOff_frac;
	int                  m_maxBandwidth;
	FactorizationMethod  m_factMethod;
	PreconditionerType   m_precondType;
	bool                 m_safeFactorization;
	bool                 m_variableBandwidth;
	bool                 m_trackReordering;

	bool                 m_setupDone;
	MatrixMap            m_offDiagMap;
	MatrixMap            m_WVMap;
	MatrixMap            m_typeMap;
	MatrixMap            m_bandedMatMap;
	MatrixMapF           m_scaleMap;

	// Used in variable-bandwidth method only, host versions
	IntVectorH           m_ks_host;
	IntVectorH           m_ks_row_host;
	IntVectorH           m_ks_col_host;
	IntVectorH           m_offDiagWidths_left_host;
	IntVectorH           m_offDiagWidths_right_host;
	IntVectorH           m_first_rows_host;
	IntVectorH           m_BOffsets_host;

	// Used in variable-bandwidth method only
	IntVector            m_ks;                    // All half-bandwidths
	IntVector            m_offDiagWidths_left;    // All left half-bandwidths in terms of rows
	IntVector            m_offDiagWidths_right;   // All right half-bandwidths in terms of rows
	IntVector            m_offDiagPerms_left;
	IntVector            m_offDiagPerms_right;
	IntVector            m_first_rows;
	IntVector            m_spike_ks;              // All half-bandwidths which are for spikes.
	                                              // m_spike_ks[i] = MAX ( m_ks[i] , m_ks[i+1] )
	IntVector            m_BOffsets;              // Offsets in banded-matrix B
	IntVector            m_ROffsets;              // Offsets in matrix R
	IntVector            m_WVOffsets;             // Offsets in matrix WV
	IntVector            m_compB2Offsets;         // Offsets in matrix compB2
	IntVector            m_partialBOffsets;       // Offsets in matrix partialB

	IntVector            m_optPerm;               // RCM reordering
	IntVector            m_optReordering;         // RCM reverse reordering

	IntVector            m_secondReordering;      // 2nd stage reverse reordering
	IntVector            m_secondPerm;            // 2nd stage reordering

	PrecVector           m_mc64RowScale;          // MC64 row scaling
	PrecVector           m_mc64ColScale;          // MC64 col scaling

	PrecVector           m_B;                     // banded matrix (LU factors)
	PrecVector           m_offDiags;              // contains the off-diagonal blocks of the original banded matrix
	PrecVector           m_R;                     // diagonal blocks in the reduced matrix (LU factors)

	PrecVectorH          m_offDiags_host;         // Used with second-stage reorder only, copy the offDiags in SpikeGragh
	PrecVectorH          m_WV_host;

	int                  m_k_reorder;             // bandwidth after reordering
	int                  m_k_mc64;                // bandwidth after MC64

	PrecValueType        m_dropOff_actual;        // actual dropOff fraction achieved

	GPUTimer             m_timer;
	double               m_time_reorder;          // CPU time for matrix reordering
	double               m_time_cpu_assemble;     // Time for acquiring the banded matrix and off-diagonal matrics on CPU
	double               m_time_transfer;         // Time for data transferring from CPU to GPU
	double               m_time_toBanded;         // GPU time for transformation or conversion to banded double       
	double               m_time_offDiags;         // GPU time to copy off-diagonal blocks
	double               m_time_bandLU;           // GPU time for LU factorization of banded blocks
	double               m_time_bandUL;           // GPU time for UL factorization of banded blocks
	double               m_time_assembly;         // GPU time for assembling the reduced matrix
	double               m_time_fullLU;           // GPU time for LU factorization of reduced matrix
	double               m_time_shuffle;          // cumulative GPU time for permutation and scaling

	template <typename Matrix>
	void transformToBandedMatrix(const Matrix&  A);

	template <typename Matrix>
	void convertToBandedMatrix(const Matrix&  A);

	void extractOffDiagonal(PrecVector& mat_WV);

	void partBandedLU();
	void partBandedLU_const();
	void partBandedLU_one();
	void partBandedLU_var();
	void partBandedUL(PrecVector& B);

	void partBandedFwdSweep(PrecVector& v);
	void partBandedFwdSweep_const(PrecVector& v);
	void partBandedFwdSweep_var(PrecVector& v);
	void partBandedBckSweep(PrecVector& v);
	void partBandedBckSweep_const(PrecVector& v);
	void partBandedBckSweep_var(PrecVector& v);

	void partFullLU();
	void partFullLU_const();
	void partFullLU_var();

	void partFullFwdSweep(PrecVector& v);
	void partFullBckSweep(PrecVector& v);
	void purifyRHS(PrecVector& v, PrecVector& res);

	void calculateSpikes(PrecVector& WV);
	void calculateSpikes_const(PrecVector& WV);
	void calculateSpikes_var(PrecVector& WV);
	void calculateSpikes_var_old(PrecVector& WV);
	void calculateSpikes(PrecVector& B2, PrecVector& WV);

	int adjustNumThreads(int inNumThreads);


	void assembleReducedMat(PrecVector& WV);

	void copyLastPartition(PrecVector& B2);

	void leftTrans(PrecVector& v, PrecVector& z);
	void rightTrans(PrecVector& v, PrecVector& z);
	void permute(PrecVector& v, IntVector& perm, PrecVector& w);
	void permuteAndScale(PrecVector& v, IntVector& perm, PrecVector& scale, PrecVector& w);
	void scaleAndPermute(PrecVector& v, IntVector& perm, PrecVector& scale, PrecVector& w);

	void combinePermutation(IntVector& perm, IntVector& perm2, IntVector& finalPerm);
	void getSRev(PrecVector& rhs, PrecVector& sol);


	bool hasZeroPivots(const PrecVectorIterator& start_B,
	                   const PrecVectorIterator& end_B,
	                   int                       k,
	                   PrecValueType             threshold);
};



/**
 * This is the constructor for the Precond class.
 */
template <typename PrecVector>
Precond<PrecVector>::Precond(int                 numPart,
                             bool                reorder,
                             bool                doMC64,
                             bool                scale,
                             double              dropOff_frac,
                             int                 maxBandwidth,
                             FactorizationMethod factMethod,
                             PreconditionerType  precondType,
                             bool                safeFactorization,
                             bool                variableBandwidth,
                             bool                trackReordering)
:	m_numPartitions(numPart),
	m_reorder(reorder),
	m_doMC64(doMC64),
	m_scale(scale),
	m_dropOff_frac((PrecValueType)dropOff_frac),
	m_maxBandwidth(maxBandwidth),
	m_factMethod(factMethod),
	m_precondType(precondType),
	m_safeFactorization(safeFactorization),
	m_variableBandwidth(variableBandwidth),
	m_trackReordering(trackReordering),
	m_setupDone(false),
	m_k_reorder(0),
	m_k_mc64(0),
	m_dropOff_actual(0),
	m_time_reorder(0),
	m_time_cpu_assemble(0),
	m_time_transfer(0),
	m_time_toBanded(0),
	m_time_offDiags(0),
	m_time_bandLU(0),
	m_time_bandUL(0),
	m_time_assembly(0),
	m_time_fullLU(0),
	m_time_shuffle(0)
{
}

/**
 * This is the default constructor for the Precond class.
 */
template <typename PrecVector>
Precond<PrecVector>::Precond()
:	m_setupDone(false),
	m_reorder(false),
	m_doMC64(false),
	m_scale(false),
	m_k_reorder(0),
	m_k_mc64(0),
	m_dropOff_actual(0),
	m_maxBandwidth(std::numeric_limits<int>::max()),
	m_time_reorder(0),
	m_time_cpu_assemble(0),
	m_time_transfer(0),
	m_time_toBanded(0),
	m_time_offDiags(0),
	m_time_bandLU(0),
	m_time_bandUL(0),
	m_time_assembly(0),
	m_time_fullLU(0),
	m_time_shuffle(0)
{
}

/**
 * This is the copy constructor for the Precond class.
 */
template <typename PrecVector>
Precond<PrecVector>::Precond(const Precond<PrecVector> &prec)
:	m_setupDone(false),
	m_k_reorder(0),
	m_k_mc64(0),
	m_dropOff_actual(0),
	m_time_reorder(0),
	m_time_cpu_assemble(0),
	m_time_transfer(0),
	m_time_toBanded(0),
	m_time_offDiags(0),
	m_time_bandLU(0),
	m_time_bandUL(0),
	m_time_assembly(0),
	m_time_fullLU(0),
	m_time_shuffle(0)
{
	m_numPartitions     = prec.m_numPartitions;

	m_reorder           = prec.m_reorder;
	m_doMC64            = prec.m_doMC64;
	m_scale             = prec.m_scale;
	m_dropOff_frac      = prec.m_dropOff_frac;
	m_maxBandwidth      = prec.m_maxBandwidth;
	m_factMethod        = prec.m_factMethod;
	m_precondType       = prec.m_precondType;
	m_safeFactorization = prec.m_safeFactorization;
	m_variableBandwidth = prec.m_variableBandwidth;
	m_trackReordering   = prec.m_trackReordering;
}

template <typename PrecVector>
Precond<PrecVector>& 
Precond<PrecVector>::operator=(const Precond<PrecVector>& prec)
{
	m_numPartitions     = prec.m_numPartitions;

	m_reorder           = prec.m_reorder;
	m_doMC64            = prec.m_doMC64;
	m_scale             = prec.m_scale;
	m_dropOff_frac      = prec.m_dropOff_frac;
	m_maxBandwidth      = prec.m_maxBandwidth;
	m_factMethod        = prec.m_factMethod;
	m_precondType       = prec.m_precondType;
	m_safeFactorization = prec.m_safeFactorization;
	m_variableBandwidth = prec.m_variableBandwidth;
	m_trackReordering   = prec.m_trackReordering;

	m_setupDone         = false;
	m_ks_host           = prec.m_ks_host;
	m_offDiagWidths_left_host = prec.m_offDiagWidths_left_host;
	m_offDiagWidths_right_host = prec.m_offDiagWidths_right_host;
	m_first_rows_host   = prec.m_first_rows_host;
	m_BOffsets_host     = prec.m_BOffsets_host;

	m_time_shuffle = 0;
	return *this;
}


/*! \brief This function updates the banded matrix and off-diagonal matrices
 *         based on the given entries.
 *
 * Assume we are to solve many systems with exactly the same matrix pattern.
 * When we have solved one, next time we don't bother doing permutation and 
 * scaling again. Instead, we keep track of the mapping from the sparse matrix
 * to the banded ones and directly update them. This function is called when
 * the solver has solved at least one system and during setup, the mapping is
 * tracked. Otherwise report error and exit.
 */
template <typename PrecVector>
void
Precond<PrecVector>::update(const PrecVector& entries)
{
	// If setup function is not called at all, directly return from this function
	if (!m_setupDone)
		throw system_error(system_error::Illegal_update, "Illegal call to update() before setup().");

	if (!m_trackReordering)
		throw system_error(system_error::Illegal_update, "Illegal call to update() with reordering tracking disabled.");

	m_time_reorder = 0.0;

	m_timer.Start();


	cusp::blas::fill(m_B, (PrecValueType) 0);

	thrust::scatter_if(
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.begin(), m_scaleMap.begin())), Multiplier<PrecValueType>()),
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.end(), m_scaleMap.end())), Multiplier<PrecValueType>()),
			m_bandedMatMap.begin(),
			m_typeMap.begin(),
			m_B.begin()
			);
	m_timer.Stop();
	m_time_cpu_assemble = m_timer.getElapsed();

	m_time_transfer = 0.0;

	////cusp::io::write_matrix_market_file(m_B, "B.mtx");
	if (m_k == 0)
		return;


	// If we are using a single partition, perform the LU factorization
	// of the banded matrix and return.
	if (m_precondType == Block || m_numPartitions == 1) {

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
	PrecVector mat_WV;
	mat_WV.resize(2 * m_k * m_k * (m_numPartitions-1));
	cusp::blas::fill(m_offDiags, (PrecValueType) 0);

	m_timer.Start();

	thrust::scatter_if(
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.begin(), m_scaleMap.begin())), Multiplier<PrecValueType>()),
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.end(), m_scaleMap.end())), Multiplier<PrecValueType>()),
			m_offDiagMap.begin(),
			m_typeMap.begin(),
			m_offDiags.begin(),
			thrust::logical_not<int>()
			);

	thrust::scatter_if(
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.begin(), m_scaleMap.begin())), Multiplier<PrecValueType>()),
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.end(), m_scaleMap.end())), Multiplier<PrecValueType>()),
			m_WVMap.begin(),
			m_typeMap.begin(),
			mat_WV.begin(),
			thrust::logical_not<int>()
			);
	m_timer.Stop();
	m_time_offDiags = m_timer.getElapsed();


	switch (m_factMethod) {
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
			PrecVector B2 = m_B;

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

/**
 * This function performs the initial preconditioner setup, based on the
 * specified matrix:
 * (1) Reorder the matrix (MC64 and/or RCM)
 * (2) Element drop-off (optional)
 * (3) LU factorization
 * (4) Get the reduced matrix
 */
template <typename PrecVector>
template <typename Matrix>
void
Precond<PrecVector>::setup(const Matrix&  A)
{
	m_n = A.num_rows;

	m_setupDone = true;

	// Form the banded matrix based on the specified matrix, either through
	// transformation (reordering and drop-off) or straight conversion.
	if (m_reorder)
		transformToBandedMatrix(A);
	else
		convertToBandedMatrix(A);

	////cusp::io::write_matrix_market_file(m_B, "B.mtx");
	if (m_k == 0)
		return;


	// If we are using a single partition, perform the LU factorization
	// of the banded matrix and return.
	if (m_precondType == Block || m_numPartitions == 1) {

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
	PrecVector mat_WV;
	mat_WV.resize(2 * m_k * m_k * (m_numPartitions-1));

	m_timer.Start();
	extractOffDiagonal(mat_WV);
	m_timer.Stop();
	m_time_offDiags = m_timer.getElapsed();


	switch (m_factMethod) {
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
			PrecVector B2 = m_B;

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
}

/**
 * This function solves the system Mz=v, for a specified vector v, where M is
 * the implicitly defined preconditioner matrix.
 */
template <typename PrecVector>
void
Precond<PrecVector>::solve(PrecVector&  v,
                           PrecVector&  z)
{
	if (m_reorder) {
		leftTrans(v, z);
		static PrecVector buffer;
		buffer.resize(m_n);
		getSRev(z, buffer);
		rightTrans(buffer, z);
	} else {
		cusp::blas::copy(v, z);
		PrecVector buffer = z;
		getSRev(buffer, z);
	}
}

/**
 * This function gets a rough solution of the input RHS.
 */
template <typename PrecVector>
void
Precond<PrecVector>::getSRev(PrecVector&  rhs,
                             PrecVector&  sol)
{
	if (m_k == 0) {
		thrust::transform(rhs.begin(), rhs.end(), m_B.begin(), sol.begin(), thrust::divides<PrecValueType>());
		return;
	}

	if (m_numPartitions > 1 && m_precondType == Spike) {
		if (!m_variableBandwidth) {
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
			static PrecVector buffer;
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

/**
 * This function left transforms the system. We first apply the MC64 row
 * scaling and permutation (or only the MC64 row permutation) after which we
 * apply the RCM row permutation.
 */
template <typename PrecVector>
void
Precond<PrecVector>::leftTrans(PrecVector&  v,
                               PrecVector&  z)
{
	if (m_scale)
		scaleAndPermute(v, m_optPerm, m_mc64RowScale, z);
	else
		permute(v, m_optPerm, z);
}

/**
 * This function right transforms the system. We apply the RCM column 
 * permutation and, if needed, the MC64 column scaling.
 */
template <typename PrecVector>
void
Precond<PrecVector>::rightTrans(PrecVector&  v,
                                PrecVector&  z)
{
	if (m_scale)
		permuteAndScale(v, m_optReordering, m_mc64ColScale, z);
	else
		permute(v, m_optReordering, z);
}

/**
 * This function transforms the input vector 'v' into the output vector 'w' by
 * applying the permutation 'perm'.
 */
template <typename PrecVector>
void
Precond<PrecVector>::permute(PrecVector&   v,
                             IntVector&    perm,
                             PrecVector&   w)
{
	m_timer.Start();
	thrust::scatter(v.begin(), v.end(), perm.begin(), w.begin());
	m_timer.Stop();
	m_time_shuffle += m_timer.getElapsed();
}

/**
 * This function transforms the input vector 'v' into the output vector 'w' by
 * applying the permutation 'perm' followed by the scaling 'scale'.
 */
template <typename PrecVector>
void
Precond<PrecVector>::permuteAndScale(PrecVector&   v,
                                     IntVector&    perm,
                                     PrecVector&   scale,
                                     PrecVector&   w)
{
	m_timer.Start();

	thrust::scatter(
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.begin(), thrust::make_permutation_iterator(scale.begin(), perm.begin()))), Multiplier<PrecValueType>()),
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.end(), thrust::make_permutation_iterator(scale.end(), perm.end()))), Multiplier<PrecValueType>()),
			perm.begin(),
			w.begin()
			);

	m_timer.Stop();
	m_time_shuffle += m_timer.getElapsed();
}

/**
 * This function transforms the input vector 'v' into the output vector 'w' by
 * applying the scaling 'scale' followed by the permutation 'perm'.
 */ 
template <typename PrecVector>
void
Precond<PrecVector>::scaleAndPermute(PrecVector&   v,
                                     IntVector&    perm,
                                     PrecVector&   scale,
                                     PrecVector&   w)
{
	m_timer.Start();
	thrust::scatter(
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.begin(), scale.begin())), Multiplier<PrecValueType>()),
			thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.end(), scale.end())), Multiplier<PrecValueType>()),
			perm.begin(),
			w.begin()
			);
	m_timer.Stop();
	m_time_shuffle += m_timer.getElapsed();
}

/**
 * This function combines two permutations to one.
 */
template <typename PrecVector>
void 
Precond<PrecVector>::combinePermutation(IntVector&  perm,
                                        IntVector&  perm2,
                                        IntVector&  finalPerm)
{
	m_timer.Start();
	thrust::gather(perm.begin(), perm.end(), perm2.begin(), finalPerm.begin());
	m_timer.Stop();
	m_time_shuffle += m_timer.getElapsed();
}


/**
 * This function applies the reordering and element drop-off algorithms to
 * obtain the banded matrix for the Spike method. On return, the following
 * member variables are set:
 *   m_B
 *       banded matrix after reordering and drop-off. This matrix is stored
 *       column-wise, band after band, in a contiguous 1-D array.
 *   m_k
 *       half band-width of the matrix m_B (after reordering and drop-off)
 *   m_optReordering
 *   m_optPerm
 *       permutation arrays obtained from the symmetric RCM algorithm
 *       row and column permutations obtained from the MC64 algorithm
 *   mc64RowScale
 *   mc64ColScale
 *       row and column scaling factors obtained from the MC64 algorithm
 */
template <typename PrecVector>
template <typename Matrix>
void
Precond<PrecVector>::transformToBandedMatrix(const Matrix&  A)
{
	CPUTimer reorder_timer, assemble_timer, transfer_timer;

	transfer_timer.Start();

	// Reorder the matrix and apply drop-off. For this, we convert the
	// input matrix to COO format and copy it on the host.
	PrecMatrixCooH Acoo = A;
	transfer_timer.Stop();
	m_time_transfer = transfer_timer.getElapsed();

	PrecVectorH  B;
	IntVectorH   optReordering;
	IntVectorH   optPerm;
	IntVectorH   mc64RowPerm;
	PrecVectorH  mc64RowScale;
	PrecVectorH  mc64ColScale;
	IntVectorH   secondReorder;
	IntVectorH   secondPerm;

	Graph<PrecValueType>  graph(m_trackReordering);

	IntVectorH   offDiagPerms_left;
	IntVectorH   offDiagPerms_right;

	MatrixMapH   offDiagMap;
	MatrixMapH   WVMap;
	MatrixMapH   typeMap;
	MatrixMapH   bandedMatMap;
	MatrixMapFH  scaleMap;


	reorder_timer.Start();
	m_k_reorder = graph.reorder(Acoo, m_doMC64, m_scale, optReordering, optPerm, mc64RowPerm, mc64RowScale, mc64ColScale, scaleMap, m_k_mc64);
	reorder_timer.Stop();

	m_time_reorder += reorder_timer.getElapsed();
	
	int dropped = 0;

	if (m_k_reorder > m_maxBandwidth || m_dropOff_frac > 0)
		dropped = graph.dropOff(m_dropOff_frac, m_maxBandwidth, m_dropOff_actual);
	else
		m_dropOff_actual = 0;

	// FIXME: this is a little bit problematic when for some off-diagonals, there is no element at all.
	m_k = m_k_reorder - dropped;


	// Verify that the required number of partitions is consistent with the
	// problem size and half-bandwidth.  If 'n' is the smallest partition size,
	// the following condition must be satisfied:
	//   K+1 <= n   (for Spike algorithm)
	// These imply a maximum allowable number of partitions.
	int maxNumPartitions = std::max(m_n / (m_k + 1), 1);
	m_numPartitions = std::min(m_numPartitions, maxNumPartitions);

	// If there is just one partition, force using constant bandwidth method.
	if (m_numPartitions == 1 || m_k == 0)
		m_variableBandwidth = false;


	// Assemble the banded matrix.
	if (m_variableBandwidth) {
		assemble_timer.Start();
		graph.assembleOffDiagMatrices(m_k, m_numPartitions, m_WV_host, m_offDiags_host, m_offDiagWidths_left_host, m_offDiagWidths_right_host, offDiagPerms_left, offDiagPerms_right, typeMap, offDiagMap, WVMap);
		assemble_timer.Stop();
		m_time_cpu_assemble += assemble_timer.getElapsed();

		reorder_timer.Start();
		graph.secondLevelReordering(m_k, m_numPartitions, secondReorder, secondPerm, m_first_rows_host);
		reorder_timer.Stop();
		m_time_reorder += reorder_timer.getElapsed();

		assemble_timer.Start();
		graph.assembleBandedMatrix(m_k, m_numPartitions, m_ks_col_host, m_ks_row_host, B,
		                           m_ks_host, m_BOffsets_host, 
		                           typeMap, bandedMatMap);
		assemble_timer.Stop();
		m_time_cpu_assemble += assemble_timer.getElapsed();
	} else {
		assemble_timer.Start();
		graph.assembleBandedMatrix(m_k, m_ks_col_host, m_ks_row_host, B, typeMap, bandedMatMap);
		assemble_timer.Stop();
		m_time_cpu_assemble += assemble_timer.getElapsed();
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

	if (m_variableBandwidth) {
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

	if (m_variableBandwidth) {
		IntVector buffer2(m_n);
		m_secondReordering = secondReorder;
		combinePermutation(m_secondReordering, m_optReordering, buffer2);
		m_optReordering = buffer2;

		m_secondPerm = secondPerm;
		combinePermutation(m_optPerm, m_secondPerm, buffer2);
		m_optPerm = buffer2;

		m_first_rows = m_first_rows_host;
	}

	{
		IntVector buffer = mc64RowPerm, buffer2(m_n);
		combinePermutation(buffer, m_optPerm, buffer2);
		m_optPerm = buffer2;
	}

	if (m_trackReordering) {
		m_offDiagMap   = offDiagMap;
		m_WVMap        = WVMap;
		m_typeMap      = typeMap;
		m_bandedMatMap = bandedMatMap;
		m_scaleMap     = scaleMap;
	}

	transfer_timer.Stop();
	m_time_transfer += transfer_timer.getElapsed();
}

/**
 * This function converts the specified sparse format matrix to a banded matrix
 * m_B which is stored column-wise, band after band, in a contiguous 1-D array.
 * It also sets m_k to be the half-bandwidth of m_B.
 */
template <typename PrecVector>
template <typename Matrix>
void
Precond<PrecVector>::convertToBandedMatrix(const Matrix&  A)
{
	// Convert matrix to COO format.
	PrecMatrixCoo Acoo = A;
	int n = Acoo.num_rows;
	int nnz = Acoo.num_entries;

	// Calculate bandwidth. Note that we use an explicit code block so
	// that the temporary array 'buffer' is freed before we resize m_B
	// (otherwise, we may run out of memory).
	{
		IntVector  buffer(nnz);
		cusp::blas::axpby(Acoo.row_indices, Acoo.column_indices, buffer, 1, -1);
		m_k = cusp::blas::nrmmax(buffer);
	}

	// Verify that the required number of partitions is consistent with the
	// problem size and half-bandwidth.  If 'n' is the smallest partition size,
	// the following two conditions must be satisfied:
	//   (1)  K+1 <= n   (for Spike algorithm)
	//   (2)  2*K <= n   (for current implementation of UL)
	// These imply a maximum allowable number of partitions.
	int  maxNumPartitions = std::max(1, m_n / std::max(m_k + 1, 2 * m_k));
	m_numPartitions = std::min(m_numPartitions, maxNumPartitions);

	// If there is just one partition, force using constant-bandwidth method.
	if (m_numPartitions == 1)
		m_variableBandwidth = false;

	// Set the size and load the banded matrix into m_B.
	m_B.resize((2*m_k+1)*n);

	int blockX = nnz, gridX = 1, gridY = 1;
	kernelConfigAdjust(blockX, gridX, gridY, BLOCK_SIZE, MAX_GRID_DIMENSION);
	dim3 grids(gridX, gridY);

	int*           d_rows = thrust::raw_pointer_cast(&(Acoo.row_indices[0]));
	int*           d_cols = thrust::raw_pointer_cast(&(Acoo.column_indices[0]));
	PrecValueType* d_vals = thrust::raw_pointer_cast(&(Acoo.values[0]));
	PrecValueType* dB     = thrust::raw_pointer_cast(&m_B[0]);

	m_timer.Start();
	device::copyFromCOOMatrixToBandedMatrix<<<grids, blockX>>>(nnz, m_k, d_rows, d_cols, d_vals, dB);
	m_timer.Stop();
	m_time_toBanded = m_timer.getElapsed();
}


/**
 * This function extracts and saves the off-diagonal blocks. Simultaneously,
 * it also initializes the specified WV matrix with the off-diagonal blocks
 * (this will be later processed to obtain the actual spike blocks in WV).
 */
template <typename PrecVector>
void
Precond<PrecVector>::extractOffDiagonal(PrecVector&  mat_WV)
{
	// If second-level reordering is enabled, the off-diagonal matrices are already in the host.
	if (m_variableBandwidth) {
		mat_WV = m_WV_host;
		m_offDiags = m_offDiags_host;
		return;
	}

	m_offDiags.resize(2 * m_k * m_k * (m_numPartitions - 1));

	PrecValueType* p_B        = thrust::raw_pointer_cast(&m_B[0]);
	PrecValueType* p_WV       = thrust::raw_pointer_cast(&mat_WV[0]);
	PrecValueType* p_offDiags = thrust::raw_pointer_cast(&m_offDiags[0]);
	int*           p_ks       = thrust::raw_pointer_cast(&m_ks[0]);

	int  partSize  = m_n / m_numPartitions;
	int  remainder = m_n % m_numPartitions;

	dim3 grids(m_k, m_numPartitions-1);

	if (m_k > 1024)
		device::copydWV_general<PrecValueType><<<grids, 512>>>(m_k, p_B, p_WV, p_offDiags, partSize, m_numPartitions, remainder);
	else if (m_k > 32)
		device::copydWV_g32<PrecValueType><<<grids, m_k>>>(m_k, p_B, p_WV, p_offDiags, partSize, m_numPartitions, remainder);
	else
		device::copydWV<PrecValueType><<<m_numPartitions-1, m_k*m_k>>>(m_k, p_B, p_WV, p_offDiags, partSize, m_numPartitions, remainder);
}

/*! \brief This function will call either Precond::partFullLU_const() or
 *		   Precond::partFullLU_var()
 *
 * This function performs the in-place LU factorization of the diagonal blocks
 * of the reduced matrix R. We take advantage of the special block structure of
 * each individual 2*k by 2*k diagonal block, namely:
 *       [ I_k  |   V  ]
 * R_i = [------+ -----]
 *       [  W   |  I_k ]
 */
template <typename PrecVector>
void
Precond<PrecVector>::partFullLU()
{
	if (!m_variableBandwidth)
		partFullLU_const();
	else
		partFullLU_var();
}


template <typename PrecVector>
void
Precond<PrecVector>::partFullLU_const()
{
	PrecValueType* d_R = thrust::raw_pointer_cast(&m_R[0]);
	int two_k = 2 * m_k;

	// The first k rows of each diagonal block do not need a division step and
	// always use a pivot = 1.
	{
		dim3 grids(m_k, m_numPartitions-1);

		if( m_k > 1024)
			device::fullLU_sub_spec_general<PrecValueType><<<grids, 512>>>(d_R, two_k, m_k);
		else
			device::fullLU_sub_spec<PrecValueType><<<grids, m_k>>>(d_R, two_k, m_k);
	}

	// The following k rows of each diagonal block require first a division by
	// the pivot.
	if (m_safeFactorization) {
		for(int i = m_k; i < two_k-1; i++) {
			int  threads = two_k-1-i;
			dim3 grids(two_k-1-i, m_numPartitions-1);

			if(threads > 1024) {
				device::fullLU_div_safe_general<PrecValueType><<<m_numPartitions-1, 512>>>(d_R, m_k, two_k, i);
				device::fullLU_sub_general<PrecValueType><<<grids, 512>>>(d_R, m_k, two_k, i);
			} else {
				device::fullLU_div_safe<PrecValueType><<<m_numPartitions-1, threads>>>(d_R, two_k, i);
				device::fullLU_sub<PrecValueType><<<grids, threads>>>(d_R, two_k, i);
			}
		}
	} else {
		for(int i = m_k; i < two_k-1; i++) {
			int  threads = two_k-1-i;
			dim3 grids(two_k-1-i, m_numPartitions-1);

			if(threads > 1024) {
				device::fullLU_div_general<PrecValueType><<<m_numPartitions-1, 512>>>(d_R, m_k, two_k, i);
				device::fullLU_sub_general<PrecValueType><<<grids, 512>>>(d_R, m_k, two_k, i);
			} else {
				device::fullLU_div<PrecValueType><<<m_numPartitions-1, threads>>>(d_R, two_k, i);
				device::fullLU_sub<PrecValueType><<<grids, threads>>>(d_R, two_k, i);
			}
		}
	}
}

template <typename PrecVector>
void
Precond<PrecVector>::partFullLU_var()
{
	PrecValueType* d_R        = thrust::raw_pointer_cast(&m_R[0]);
	int*           p_spike_ks = thrust::raw_pointer_cast(&m_spike_ks[0]);
	int*           p_ROffsets = thrust::raw_pointer_cast(&m_ROffsets[0]);
	
	int        two_k = 2 * m_k;

	// The first k rows of each diagonal block do not need a division step and
	// always use a pivot = 1.
	{
		dim3 grids(m_k, m_numPartitions-1);

		if( m_k > 1024)
			device::var::fullLU_sub_spec_general<PrecValueType><<<grids, 512>>>(d_R, p_spike_ks, p_ROffsets);
		else
			device::var::fullLU_sub_spec<PrecValueType><<<grids, m_k>>>(d_R, p_spike_ks, p_ROffsets);
	}

	// The following k rows of each diagonal block require first a division by
	// the pivot.
	if (m_safeFactorization) {
		for(int i = m_k; i < two_k-1; i++) {
			int  threads = two_k-1-i;
			dim3 grids(two_k-1-i, m_numPartitions-1);

			if(threads > 1024) {
				device::var::fullLU_div_safe_general<PrecValueType><<<m_numPartitions-1, 512>>>(d_R, p_spike_ks, p_ROffsets, i);
				device::var::fullLU_sub_general<PrecValueType><<<grids, 512>>>(d_R, p_spike_ks, p_ROffsets, i);
			} else {
				device::var::fullLU_div_safe<PrecValueType><<<m_numPartitions-1, threads>>>(d_R, p_spike_ks, p_ROffsets, i);
				device::var::fullLU_sub<PrecValueType><<<grids, threads>>>(d_R, p_spike_ks, p_ROffsets, i);
			}
		}
	} else {
		for(int i = m_k; i < two_k-1; i++) {
			int  threads = two_k-1-i;
			dim3 grids(two_k-1-i, m_numPartitions-1);

			if(threads > 1024) {
				device::var::fullLU_div_general<PrecValueType><<<m_numPartitions-1, 512>>>(d_R, p_spike_ks,  p_ROffsets, i);
				device::var::fullLU_sub_general<PrecValueType><<<grids, 512>>>(d_R, p_spike_ks,  p_ROffsets, i);
			} else {
				device::var::fullLU_div<PrecValueType><<<m_numPartitions-1, threads>>>(d_R, p_spike_ks,  p_ROffsets, i);
				device::var::fullLU_sub<PrecValueType><<<grids, threads>>>(d_R, p_spike_ks,  p_ROffsets, i);
			}
		}
	}

	{
		dim3 grids(m_k-1, m_numPartitions-1);
		if (m_k >= 1024)
			device::var::fullLU_post_divide_general<PrecValueType><<<grids, 512>>>(d_R, p_spike_ks, p_ROffsets);
		else
			device::var::fullLU_post_divide<PrecValueType><<<grids, m_k-1>>>(d_R, p_spike_ks, p_ROffsets);
	}
}

/*! \brief This function will call Precond::partBandedLU_one(), 
 * Precond::partBandedLU_const() or Precond::partBandedLU_var().
 *
 * This function performs the in-place LU factorization of the diagonal blocks
 * of the specified banded matrix B, on a per-partition basis, using the
 * "window sliding" method.
 */
template <typename PrecVector>
void
Precond<PrecVector>::partBandedLU()
{
	if (m_variableBandwidth) {
		// Variable bandwidth method. Note that in this situation, there
		// must be more than one partition.
		partBandedLU_var();
	} else {
		// Constant bandwidth method.
		if (m_numPartitions > 1)
			partBandedLU_const();
		else
			partBandedLU_one();
	}
}

template <typename PrecVector>
void
Precond<PrecVector>::partBandedLU_one()
{
	// As the name implies, this function can only be called if we arte using a single
	// partition. In this case, the entire banded matrix m_B is LU factorized.

	PrecValueType* dB = thrust::raw_pointer_cast(&m_B[0]);

	if (m_ks_col_host.size() < m_n)
		m_ks_col_host.resize(m_n, m_k);

	if (m_ks_row_host.size() < m_n)
		m_ks_row_host.resize(m_n, m_k);

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
					device::bandLU_critical_div_onePart_safe_general<PrecValueType><<<1, 512>>>(dB, st_row, m_k, threadsNum);
				else
					device::bandLU_critical_div_onePart_general<PrecValueType><<<threadsNum/512+1, 512>>>(dB, st_row, m_k, threadsNum);

				device::bandLU_critical_sub_onePart_general<PrecValueType><<<blockX, 512>>>(dB, st_row, m_k, threadsNum);
			} else {
				if (m_safeFactorization)
					device::bandLU_critical_div_onePart_safe<PrecValueType><<<1, threadsNum>>>(dB, st_row, m_k);
				else
					device::bandLU_critical_div_onePart<PrecValueType><<<1, threadsNum>>>(dB, st_row, m_k);

				device::bandLU_critical_sub_onePart<PrecValueType><<<blockX, threadsNum>>>(dB, st_row, m_k);
			}
		}
	} else if (m_k > 27) {
		if (m_safeFactorization)
			device::bandLU_g32_safe<PrecValueType><<<1, 512>>>(dB, m_k, m_n, 0);
		else
			device::bandLU_g32<PrecValueType><<<1, 512>>>(dB, m_k, m_n, 0);
	} else {
		if (m_safeFactorization)
			device::bandLU_safe<PrecValueType><<<1,  m_k * m_k>>>(dB, m_k, m_n, 0);
		else
			device::bandLU<PrecValueType><<<1,  m_k * m_k>>>(dB, m_k, m_n, 0);
			////device::swBandLU<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
	}


	if (m_safeFactorization)
		device::boostLastPivot<PrecValueType><<<1, 1>>>(dB, m_n, m_k, m_n, 0);


	// If not using safe factorization, check the factorized banded matrix for any
	// zeros on its diagonal (this means a zero pivot).
	if (!m_safeFactorization && hasZeroPivots(m_B.begin(), m_B.end(), m_k, (PrecValueType) BURST_VALUE))
		throw system_error(system_error::Zero_pivoting, "Found a pivot equal to zero (partBandedLU_one).");


	int gridX = m_n, gridY = 1;
	kernelConfigAdjust(gridX, gridY, MAX_GRID_DIMENSION);
	dim3 grids(gridX, gridY);
	if (m_k > 1024)
		device::bandLU_post_divide_general<PrecValueType><<<grids, 512>>>(dB, m_k, m_n);
	else
		device::bandLU_post_divide<PrecValueType><<<grids, m_k>>>(dB, m_k, m_n);
}

template <typename PrecVector>
void
Precond<PrecVector>::partBandedLU_const()
{
	// Note that this function is called only if there are two or more partitions.
	// Moreover, if the factorization method is LU_only, all diagonal blocks in
	// each partition are LU factorized. If the method is LU_UL, then the diagonal
	// block in the last partition is *not* factorized.

	PrecValueType* dB = thrust::raw_pointer_cast(&m_B[0]);

	int n_eff = m_n;
	int numPart_eff = m_numPartitions;

	if (m_factMethod == LU_UL && m_numPartitions > 1 && m_precondType != Block) {
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
						device::bandLU_critical_div_safe_general<PrecValueType><<<remainder, 512>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandLU_critical_div_general<PrecValueType><<<remainder, 512>>>(dB, st_row, m_k, partSize, remainder);
					dim3 tmpGrids(threadsNum, remainder);
					device::bandLU_critical_sub_general<PrecValueType><<<tmpGrids, 512>>>(dB, st_row, m_k, partSize, remainder);
				} else {
					if (m_safeFactorization)
						device::bandLU_critical_div_safe<PrecValueType><<<remainder, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandLU_critical_div<PrecValueType><<<remainder, threadsNum>>>(dB, st_row, m_k, partSize, remainder);

					dim3 tmpGrids(threadsNum, remainder);
					device::bandLU_critical_sub<PrecValueType><<<tmpGrids, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
				}
			} else {
				threadsNum = m_k;
				if (threadsNum > final_partition_size - st_row - 1)
					threadsNum = final_partition_size - st_row - 1;
				if (threadsNum > 1024) {
					if (m_safeFactorization)
						device::bandLU_critical_div_safe_general<PrecValueType><<<numPart_eff, 512>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandLU_critical_div_general<PrecValueType><<<numPart_eff, 512>>>(dB, st_row, m_k, partSize, remainder);

					dim3 tmpGrids(threadsNum, numPart_eff);
					device::bandLU_critical_sub_general<PrecValueType><<<tmpGrids, 512>>>(dB, st_row, m_k, partSize, remainder);
				} else {
					if (m_safeFactorization)
						device::bandLU_critical_div_safe<PrecValueType><<<numPart_eff, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandLU_critical_div<PrecValueType><<<numPart_eff, threadsNum>>>(dB, st_row, m_k, partSize, remainder);

					dim3 tmpGrids(threadsNum, numPart_eff);
					device::bandLU_critical_sub<PrecValueType><<<tmpGrids, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
				}
			}
		}
	} else if (m_k > 27) {
		if (m_safeFactorization)
			device::bandLU_g32_safe<PrecValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder);
		else
			device::bandLU_g32<PrecValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder);
	} else {
		if (m_safeFactorization)
			device::bandLU_safe<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
		else
			device::bandLU<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
			////device::swBandLU<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
	}


	// If not using safe factorization, check the factorized banded matrix for any
	// zeros on its diagonal (this means a zero pivot). Note that we must only check
	// the diagonal blocks corresponding to the partitions for which LU was applied.
	if (!m_safeFactorization && hasZeroPivots(m_B.begin(), m_B.begin() + n_eff * (2*m_k+1), m_k, (PrecValueType) BURST_VALUE))
		throw system_error(system_error::Zero_pivoting, "Found a pivot equal to zero (partBandedLU_const).");


	if (m_numPartitions == 1) {
		int  gridX = m_n;
		int  gridY = 1;
		kernelConfigAdjust(gridX, gridY, MAX_GRID_DIMENSION);
		dim3 grids(gridX, gridY);
		if (m_k > 1024)
			device::bandLU_post_divide_general<PrecValueType><<<grids, 512>>>(dB, m_k, m_n);
		else
			device::bandLU_post_divide<PrecValueType><<<grids, m_k>>>(dB, m_k, m_n);
	}
}


template <typename PrecVector>
void
Precond<PrecVector>::partBandedLU_var()
{
	// Note that this function can only be called if there are two or more partitions.
	// Also, in this case, the factorization method is LU_only which implies that all
	// partitions are LU factorized.

	PrecValueType* dB         = thrust::raw_pointer_cast(&m_B[0]);
	int*           p_ks       = thrust::raw_pointer_cast(&m_ks[0]);
	int*           p_BOffsets = thrust::raw_pointer_cast(&m_BOffsets[0]);

	int tmp_k = cusp::blas::nrmmax(m_ks);
	int partSize  = m_n / m_numPartitions;
	int remainder = m_n % m_numPartitions;

	if(tmp_k >= CRITICAL_THRESHOLD) {
		int final_partition_size = partSize + 1;
		int blockY = 0;
		int threadsNum = adjustNumThreads(cusp::blas::nrm1(m_ks) / m_numPartitions);
		int last = 0;

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
					device::var::bandLU_critical_div_safe_general<PrecValueType><<<remainder, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder);
				else
					device::var::bandLU_critical_div_general<PrecValueType><<<remainder, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder);

				dim3 tmpGrids(blockY, remainder);
				device::var::bandLU_critical_sub_general<PrecValueType><<<tmpGrids, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder, last);
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
					device::var::bandLU_critical_div_safe_general<PrecValueType><<<m_numPartitions, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder);
				else
					device::var::bandLU_critical_div_general<PrecValueType><<<m_numPartitions, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder);

				dim3 tmpGrids(blockY, m_numPartitions);
				device::var::bandLU_critical_sub_general<PrecValueType><<<tmpGrids, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder, last);
			}
		}
	} else if (tmp_k > 27){
		if (m_safeFactorization)
			device::var::bandLU_g32_safe<PrecValueType><<<m_numPartitions, 512>>>(dB, p_ks, p_BOffsets, partSize, remainder);
		else
			device::var::bandLU_g32<PrecValueType><<<m_numPartitions, 512>>>(dB, p_ks, p_BOffsets, partSize, remainder);
	} else {
		if (m_safeFactorization)
			device::var::bandLU_safe<PrecValueType><<<m_numPartitions,  tmp_k * tmp_k >>>(dB, p_ks, p_BOffsets, partSize, remainder);
		else
			device::var::bandLU<PrecValueType><<<m_numPartitions,  tmp_k * tmp_k>>>(dB, p_ks, p_BOffsets, partSize, remainder);
	}


	if (m_safeFactorization)
		device::var::boostLastPivot<PrecValueType><<<m_numPartitions, 1>>>(dB, partSize, p_ks, p_BOffsets, partSize, remainder);


	// If not using safe factorization, check for zero pivots in the factorized banded
	// matrix, one partition at a time.
	if (!m_safeFactorization) {
		for (int i = 0; i < m_numPartitions; i++) {
			if (hasZeroPivots(m_B.begin() + m_BOffsets_host[i], m_B.begin() + m_BOffsets_host[i+1], m_ks_host[i], (PrecValueType) BURST_VALUE))
				throw system_error(system_error::Zero_pivoting, "Found a pivot equal to zero (partBandedLU_var).");
		}
	}


	int gridX = partSize+1;
	int gridY = 1;
	kernelConfigAdjust(gridX, gridY, MAX_GRID_DIMENSION);
	dim3 grids(gridX, gridY);

	for (int i=0; i<m_numPartitions ; i++) {
		if (i < remainder) {
			if (m_ks_host[i] <= 1024)
				device::var::bandLU_post_divide_per_partition<PrecValueType><<<grids, m_ks_host[i]>>>(dB, m_ks_host[i], m_BOffsets_host[i], partSize + 1);
			else
				device::var::bandLU_post_divide_per_partition_general<PrecValueType><<<grids, 512>>>(dB, m_ks_host[i], m_BOffsets_host[i], partSize + 1);
		}
		else {
			if (m_ks_host[i] <= 1024)
				device::var::bandLU_post_divide_per_partition<PrecValueType><<<grids, m_ks_host[i]>>>(dB, m_ks_host[i], m_BOffsets_host[i], partSize);
			else
				device::var::bandLU_post_divide_per_partition_general<PrecValueType><<<grids, 512>>>(dB, m_ks_host[i], m_BOffsets_host[i], partSize);
		}
	}
}

/**
 * This function performs the in-place UL factorization of the diagonal blocks
 * of the specified banded matrix B, on a per-partition basis, using the
 * "window sliding" method.
 */
template <typename PrecVector>
void
Precond<PrecVector>::partBandedUL(PrecVector& B)
{
	// Note that this function can only be called if using the constant band
	// method and there are two or more partitions.
	// In any other situation, we use LU only factorization.
	// This means that the diagonal block for the first partition is never
	// UL factorized.


	int partSize  = m_n / m_numPartitions;
	int remainder = m_n % m_numPartitions;
	int n_first = (remainder == 0 ? partSize : (partSize + 1));

	PrecValueType* dB = thrust::raw_pointer_cast(&B[(2 * m_k + 1) * n_first]);

	int n_eff = m_n - n_first;
	int numPart_eff = m_numPartitions - 1;

	partSize = n_eff / numPart_eff;
	remainder = n_eff % numPart_eff;

	if(m_k >= CRITICAL_THRESHOLD) {
		int n_final = partSize + 1;
		int threadsNum = 0;
		for (int st_row = n_final - 1; st_row > 0; st_row--) {
			if (st_row == n_final - 1) {
				if (remainder == 0) continue;
				threadsNum = m_k;
				if(st_row < m_k)
					threadsNum = st_row;
				dim3 tmpGrids(threadsNum, remainder);
				if (threadsNum > 1024) {
					if (m_safeFactorization)
						device::bandUL_critical_div_safe_general<PrecValueType><<<remainder, 512>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandUL_critical_div_general<PrecValueType><<<remainder, 512>>>(dB, st_row, m_k, partSize, remainder);
					device::bandUL_critical_sub_general<PrecValueType><<<tmpGrids, 512>>>(dB, st_row, m_k, partSize, remainder);
				} else {
					if (m_safeFactorization)
						device::bandUL_critical_div_safe<PrecValueType><<<remainder, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandUL_critical_div<PrecValueType><<<remainder, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					device::bandUL_critical_sub<PrecValueType><<<tmpGrids, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
				}
			} else {
				threadsNum = m_k;
				if(st_row < m_k)
					threadsNum = st_row;
				dim3 tmpGrids(threadsNum, numPart_eff);
				if(threadsNum > 1024) {
					if (m_safeFactorization)
						device::bandUL_critical_div_safe_general<PrecValueType> <<<numPart_eff, 512>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandUL_critical_div_general<PrecValueType> <<<numPart_eff, 512>>>(dB, st_row, m_k, partSize, remainder);
					device::bandUL_critical_sub_general<PrecValueType> <<<tmpGrids, 512>>>(dB, st_row, m_k, partSize, remainder);
				} else {
					if (m_safeFactorization)
						device::bandUL_critical_div_safe<PrecValueType> <<<numPart_eff, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					else
						device::bandUL_critical_div<PrecValueType> <<<numPart_eff, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
					device::bandUL_critical_sub<PrecValueType> <<<tmpGrids, threadsNum>>>(dB, st_row, m_k, partSize, remainder);
				}
			}
		}
	} else if (m_k > 27) {
		if (m_safeFactorization)
			device::bandUL_g32_safe<PrecValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder);
		else
			device::bandUL_g32<PrecValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder);
	} else {
		if (m_safeFactorization)
			device::bandUL_safe<PrecValueType><<<numPart_eff, m_k * m_k>>>(dB, m_k, partSize, remainder);
		else
			device::bandUL<PrecValueType><<<numPart_eff, m_k * m_k>>>(dB, m_k, partSize, remainder);
			////device::swBandUL<PrecValueType><<<numPart_eff, m_k * m_k>>>(dB, m_k, partSize, remainder);
	}


	// If not using safe factorization, check for zero pivots in the factorized
	// banded matrix.
	if (!m_safeFactorization && hasZeroPivots(B.begin() + (2 * m_k + 1) * n_first, B.end(), m_k, (PrecValueType) BURST_VALUE))
		throw system_error(system_error::Zero_pivoting, "Found a pivot equal to zero (partBandedUL).");
}


/*! \brief This function will call either Precond::partBandedFwdElim_const()
 * or Precond::partBandedFwdElim_var()
 *
 * This function performs the forward elimination sweep for the given banded
 * matrix B (assumed to encode the LU factors) and vector v.
 */
template <typename PrecVector>
void 
Precond<PrecVector>::partBandedFwdSweep(PrecVector&  v)
{
	if (!m_variableBandwidth)
		partBandedFwdSweep_const(v);
	else
		partBandedFwdSweep_var(v);
}

template <typename PrecVector>
void 
Precond<PrecVector>::partBandedFwdSweep_const(PrecVector&  v)
{
	PrecValueType* p_B = thrust::raw_pointer_cast(&m_B[0]);
	PrecValueType* p_v = thrust::raw_pointer_cast(&v[0]);

	int partSize  = m_n / m_numPartitions;
	int remainder = m_n % m_numPartitions;

	if (m_precondType == Block || m_factMethod == LU_only || m_numPartitions == 1) {
		if (m_k > 1024)
			device::forwardElimL_general<PrecValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else if (m_k > 32)
			device::forwardElimL_g32<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else
			device::forwardElimL<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
	} else {
		if (m_k > 1024)
			device::forwardElimL_LU_UL_general<PrecValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else if (m_k > 32)
			device::forwardElimL_LU_UL_g32<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else
			device::forwardElimL_LU_UL<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
	}
}

template <typename PrecVector>
void 
Precond<PrecVector>::partBandedFwdSweep_var(PrecVector&  v)
{
	PrecValueType* p_B        = thrust::raw_pointer_cast(&m_B[0]);
	PrecValueType* p_v        = thrust::raw_pointer_cast(&v[0]);
	int*           p_ks       = thrust::raw_pointer_cast(&m_ks[0]);
	int*           p_BOffsets = thrust::raw_pointer_cast(&m_BOffsets[0]);

	int tmp_k     = cusp::blas::nrmmax(m_ks);
	int partSize  = m_n / m_numPartitions;
	int remainder = m_n % m_numPartitions;

	if (tmp_k > 1024)
		device::var::fwdElim_sol<PrecValueType><<<m_numPartitions, 512>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
	else if (tmp_k > 32)
		device::var::fwdElim_sol_medium<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
	else
		device::var::fwdElim_sol_narrow<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
}

/*! \brief This function will call either Precond::partBandedBckSweep_const()
 * or Precond::partBandedBckSweep_var().
 *
 * This function performs the backward substitution sweep for the given banded
 * matrix B (assumed to encode the LU factors) and vector v.
 */
template <typename PrecVector>
void 
Precond<PrecVector>::partBandedBckSweep(PrecVector&  v)
{
	if (!m_variableBandwidth)
		partBandedBckSweep_const(v);
	else
		partBandedBckSweep_var(v);
}

template <typename PrecVector>
void 
Precond<PrecVector>::partBandedBckSweep_const(PrecVector&  v)
{
	PrecValueType* p_B = thrust::raw_pointer_cast(&m_B[0]);
	PrecValueType* p_v = thrust::raw_pointer_cast(&v[0]);

	int partSize  = m_n / m_numPartitions;
	int remainder = m_n % m_numPartitions;

	if (m_precondType == Block || m_factMethod == LU_only || m_numPartitions == 1) {
		if (m_numPartitions > 1) {
			if (m_k > 1024)
				device::backwardElimU_general<PrecValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
			else if (m_k > 32)
				device::backwardElimU_g32<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
			else
				device::backwardElimU<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		} else {
			int gridX = 1;
			int blockX = m_n;
			if (blockX > BLOCK_SIZE) {
				gridX = (blockX + BLOCK_SIZE - 1) / BLOCK_SIZE;
				blockX = BLOCK_SIZE;
			}
			dim3 grids(gridX, m_numPartitions);

			device::preBck_sol_divide<PrecValueType><<<grids, blockX>>>(m_n, m_k, p_B, p_v, partSize, remainder);

			if (m_k > 1024)
				device::bckElim_sol<PrecValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
			else if (m_k > 32)
				device::bckElim_sol_medium<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
			else
				device::bckElim_sol_narrow<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		}
	} else {
		if (m_k > 1024)
			device::backwardElimU_LU_UL_general<PrecValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else if (m_k > 32)
			device::backwardElimU_LU_UL_g32<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
		else
			device::backwardElimU_LU_UL<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
	}
}

template <typename PrecVector>
void 
Precond<PrecVector>::partBandedBckSweep_var(PrecVector&  v)
{
	PrecValueType* p_B        = thrust::raw_pointer_cast(&m_B[0]);
	PrecValueType* p_v        = thrust::raw_pointer_cast(&v[0]);
	int*           p_ks       = thrust::raw_pointer_cast(&m_ks[0]);
	int*           p_BOffsets = thrust::raw_pointer_cast(&m_BOffsets[0]);

	int tmp_k      = cusp::blas::nrmmax(m_ks);
	int partSize   = m_n / m_numPartitions;
	int remainder  = m_n % m_numPartitions;

	int gridX = 1, blockX = partSize + 1;
	kernelConfigAdjust(blockX, gridX, BLOCK_SIZE);
	dim3 grids(gridX, m_numPartitions);
	device::var::preBck_sol_divide<PrecValueType><<<grids, blockX>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);

	if (tmp_k > 1024)
		device::var::bckElim_sol<PrecValueType><<<m_numPartitions, 512>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
	else if (tmp_k > 32) 
		device::var::bckElim_sol_medium<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
	else
		device::var::bckElim_sol_narrow<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
}

/**
 * This function performs the forward elimination sweep for the given full
 * matrix R (assumed to encode the LU factors) and vector v.
 */
template <typename PrecVector>
void 
Precond<PrecVector>::partFullFwdSweep(PrecVector&  v)
{
	PrecValueType* p_R = thrust::raw_pointer_cast(&m_R[0]);
	PrecValueType* p_v = thrust::raw_pointer_cast(&v[0]);

	int partSize  = m_n / m_numPartitions;
	int remainder = m_n % m_numPartitions;

	dim3 grids(m_numPartitions-1, 1);

	if (!m_variableBandwidth) {
		if (m_k > 512)
			device::forwardElimLNormal_g512<PrecValueType><<<grids, 512>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
		else
			device::forwardElimLNormal<PrecValueType><<<grids, 2*m_k-1>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
	} else {
		int* p_ROffsets = thrust::raw_pointer_cast(&m_ROffsets[0]);
		int* p_spike_ks = thrust::raw_pointer_cast(&m_spike_ks[0]);

		if (m_k > 512)
			device::var::fwdElim_full<PrecValueType><<<grids, 512>>>(m_n, p_spike_ks,  p_ROffsets, p_R, p_v, partSize, remainder);
		else
			device::var::fwdElim_full_narrow<PrecValueType><<<grids, m_k>>>(m_n, p_spike_ks, p_ROffsets, p_R, p_v, partSize, remainder);
	}
}


/**
 * This function performs the backward substitution sweep for the given full
 * matrix R (assumed to encode the LU factors) and vector v.
 */
template <typename PrecVector>
void 
Precond<PrecVector>::partFullBckSweep(PrecVector&  v)
{
	PrecValueType* p_R = thrust::raw_pointer_cast(&m_R[0]);
	PrecValueType* p_v = thrust::raw_pointer_cast(&v[0]);

	int partSize  = m_n / m_numPartitions;
	int remainder = m_n % m_numPartitions;

	dim3 grids(m_numPartitions-1, 1);

	if (!m_variableBandwidth) {
		if (m_k > 512)
			device::backwardElimUNormal_g512<PrecValueType><<<grids, 512>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
		else
			device::backwardElimUNormal<PrecValueType><<<grids, 2*m_k-1>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
	} else {
		int* p_ROffsets = thrust::raw_pointer_cast(&m_ROffsets[0]);
		int* p_spike_ks = thrust::raw_pointer_cast(&m_spike_ks[0]);

		if (m_k > 512) {
			device::var::preBck_full_divide<PrecValueType><<<m_numPartitions-1, 512>>>(m_n, p_spike_ks, p_ROffsets, p_R, p_v, partSize, remainder);
			device::var::bckElim_full<PrecValueType><<<grids, 512>>>(m_n, p_spike_ks, p_ROffsets, p_R, p_v, partSize, remainder);
		}
		else {
			device::var::preBck_full_divide_narrow<PrecValueType><<<m_numPartitions-1, m_k>>>(m_n, p_spike_ks, p_ROffsets, p_R, p_v, partSize, remainder);
			device::var::bckElim_full_narrow<PrecValueType><<<grids, 2*m_k-1>>>(m_n, p_spike_ks, p_ROffsets, p_R, p_v, partSize, remainder);
		}
	}
}

/**
 * This function applies the purification step by performing a specialized
 * inner product between the off-diagonal blocks of the original matrix
 * and the vector 'v'. The result is stored in the output vector 'res'.
 */
template <typename PrecVector>
void 
Precond<PrecVector>::purifyRHS(PrecVector&  v,
                               PrecVector&  res)
{
	PrecValueType* p_offDiags = thrust::raw_pointer_cast(&m_offDiags[0]);
	PrecValueType* p_v        = thrust::raw_pointer_cast(&v[0]);
	PrecValueType* p_res      = thrust::raw_pointer_cast(&res[0]);

	int partSize   = m_n / m_numPartitions;
	int remainder  = m_n % m_numPartitions;

	dim3 grids(m_k, m_numPartitions-1);

	if (!m_variableBandwidth) {
		if (m_k > 256)
			device::innerProductBCX_g256<PrecValueType><<<grids, 256>>>(p_offDiags, p_v, p_res, m_n, m_k, partSize, m_numPartitions, remainder);
		else if (m_k > 64)
			device::innerProductBCX_g64<PrecValueType><<<grids, 256>>>(p_offDiags, p_v, p_res, m_n, m_k, partSize, m_numPartitions, remainder);
		else if (m_k > 32)
			device::innerProductBCX_g32<PrecValueType><<<grids, 64>>>(p_offDiags, p_v, p_res, m_n, m_k, partSize, m_numPartitions, remainder);
		else
			device::innerProductBCX<PrecValueType><<<grids, 32>>>(p_offDiags, p_v, p_res, m_n, m_k, partSize, m_numPartitions, remainder);
	} else {
		int* p_WVOffsets = thrust::raw_pointer_cast(&m_WVOffsets[0]);
		int* p_spike_ks  = thrust::raw_pointer_cast(&m_spike_ks[0]);
		
		if (m_k > 256)
			device::innerProductBCX_var_bandwidth_g256<PrecValueType><<<grids, 256>>>(p_offDiags, p_v, p_res, m_n, p_spike_ks, p_WVOffsets, partSize, m_numPartitions, remainder);
		else if (m_k > 64)
			device::innerProductBCX_var_bandwidth_g64<PrecValueType><<<grids, 256>>>(p_offDiags, p_v, p_res, m_n, p_spike_ks, p_WVOffsets, partSize, m_numPartitions, remainder);
		else if (m_k > 32)
			device::innerProductBCX_var_bandwidth_g32<PrecValueType><<<grids, 64>>>(p_offDiags, p_v, p_res, m_n, p_spike_ks, p_WVOffsets, partSize, m_numPartitions, remainder);
		else
			device::innerProductBCX_var_bandwidth<PrecValueType><<<grids, 32>>>(p_offDiags, p_v, p_res, m_n, p_spike_ks, p_WVOffsets, partSize, m_numPartitions, remainder);
	}
}

/*! \brief This function will either call Precond::calculateSpikes_const()
 * or Precond::calculateSpikes_var().
 *
 * This function calculates the spike blocks in the LU_only case.
 */
template <typename PrecVector>
void
Precond<PrecVector>::calculateSpikes(PrecVector&  WV)
{
	if (!m_variableBandwidth) {
		calculateSpikes_const(WV);
		return;
	}

	int totalRHSCount = cusp::blas::nrm1(m_offDiagWidths_right_host) + cusp::blas::nrm1(m_offDiagWidths_left_host);
	if (totalRHSCount >= 2800) {
		calculateSpikes_var(WV);
		return;
	}

	calculateSpikes_var_old(WV);
}

template <typename PrecVector>
void
Precond<PrecVector>::calculateSpikes_var_old(PrecVector&  WV)
{
	PrecVector WV_spare(m_k*m_k);

	PrecValueType* p_WV       = thrust::raw_pointer_cast(&WV[0]);
	PrecValueType* p_WV_spare = thrust::raw_pointer_cast(&WV_spare[0]);

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

		PrecVector extV(m_k * n_eff, 0), buffer;

		PrecValueType* p_extV             = thrust::raw_pointer_cast(&extV[0]);
		PrecValueType* p_B                = thrust::raw_pointer_cast(&m_B[0]);
		int*           p_secondReordering = thrust::raw_pointer_cast(&m_secondReordering[0]);
		int*           p_secondPerm       = thrust::raw_pointer_cast(&m_secondPerm[0]);

		dim3 gridsCopy(m_k, numPart_eff);
		dim3 gridsSweep(numPart_eff, m_k);

		int* p_ks                  = thrust::raw_pointer_cast(&m_ks[0]);
		int* p_offDiagWidths_right = thrust::raw_pointer_cast(&m_offDiagWidths_right[0]);
		int* p_offDiagPerms_right  = thrust::raw_pointer_cast(&m_offDiagPerms_right[0]);
		int* p_first_rows          = thrust::raw_pointer_cast(&m_first_rows[0]);
		int* p_offsets             = thrust::raw_pointer_cast(&m_BOffsets[0]);

		int permuteBlockX = n_eff;
		int permuteGridX = 1;
		kernelConfigAdjust(permuteBlockX, permuteGridX, BLOCK_SIZE);
		dim3 gridsPermute(permuteGridX, m_k);

		{
			device::copyWVFromOrToExtendedV_general<PrecValueType><<<gridsCopy, numThreadsToUse>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, false);
			buffer.resize((m_k - (BUF_FACTOR-1) * (m_k / BUF_FACTOR)) * n_eff);

			PrecValueType* p_buffer = thrust::raw_pointer_cast(&buffer[0]);

			for (int i=0; i<BUF_FACTOR; i++) {
				gridsPermute.y = m_k / BUF_FACTOR;
				if (i == BUF_FACTOR - 1)
					gridsPermute.y = m_k - (BUF_FACTOR-1) * (m_k/BUF_FACTOR);
				device::permute<PrecValueType><<<gridsPermute, permuteBlockX>>>(n_eff, p_extV+(i*(m_k/BUF_FACTOR)*n_eff), p_buffer, p_secondPerm);
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
					device::var::fwdElim_rightSpike_per_partition<PrecValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>> (n_eff, m_ks_host[i], m_BOffsets_host[i]+m_ks_host[i]+(2*m_ks_host[i]+1)*(m_first_rows_host[i]-pseudo_first_row), p_B, p_extV, m_first_rows_host[i], last_row);
					
					int blockX = last_row - m_first_rows_host[i];
					int gridX = 1;
					kernelConfigAdjust(blockX, gridX, BLOCK_SIZE);
					dim3 grids(gridX, m_offDiagWidths_right_host[i]);
					device::var::preBck_rightSpike_divide_per_partition<PrecValueType><<<grids, blockX>>> (n_eff, m_ks_host[i], m_BOffsets_host[i]+m_ks_host[i]+(2*m_ks_host[i]+1)*(m_first_rows_host[i]-pseudo_first_row), p_B, p_extV, m_first_rows_host[i], last_row);

					m_first_rows_host[i] = thrust::reduce(m_secondPerm.begin()+(last_row-m_k), m_secondPerm.begin()+last_row, last_row, thrust::minimum<int>());
					device::var::bckElim_rightSpike_per_partition<PrecValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>> (n_eff, m_ks_host[i], m_BOffsets_host[i]+m_ks_host[i]+(2*m_ks_host[i]+1)*(last_row-pseudo_first_row-1), p_B, p_extV, m_first_rows_host[i], last_row);

					pseudo_first_row = last_row;
				}
			}

			for (int i=0; i<BUF_FACTOR; i++) {
				gridsPermute.y = m_k / BUF_FACTOR;
				if (i == BUF_FACTOR - 1)
					gridsPermute.y = m_k - (BUF_FACTOR-1) * (m_k/BUF_FACTOR);
				device::permute<PrecValueType><<<gridsPermute, permuteBlockX>>>(n_eff, p_extV+(i*(m_k/BUF_FACTOR)*n_eff), p_buffer, p_secondReordering);
				thrust::copy(buffer.begin(), buffer.begin()+(gridsPermute.y * n_eff), extV.begin()+i*(m_k/BUF_FACTOR)*n_eff);
			}

			device::copyWVFromOrToExtendedV_general<PrecValueType><<<gridsCopy, numThreadsToUse>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, true);
		}
		for (int i=0; i<numPart_eff; i++) {
			cusp::blas::fill(WV_spare, (PrecValueType) 0);
			device::matrixVReordering_perPartition<PrecValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>>(m_k, p_WV+2*i*m_k*m_k, p_WV_spare, p_offDiagPerms_right+i*m_k);
			thrust::copy(WV_spare.begin(), WV_spare.end(), WV.begin() + (2*i*m_k*m_k));
		}
	}


	// Copy WV into extW, perform sweeps to calculate extW, then copy back extW to WV.
	// Note that we skip the first partition (no left spike associated with it). Also
	// note that we perform full sweeps using the L and U factors to calculate the
	// entire left spikes W.
	{
		int n_eff       = m_n - first_partition_size;
		int numPart_eff = m_numPartitions - 1;
		int partSize    = n_eff / numPart_eff;
		int remainder   = n_eff % numPart_eff;

		const int BUF_FACTOR = 16;

		PrecVector extW(m_k * n_eff, 0), buffer;

		PrecValueType* p_extW = thrust::raw_pointer_cast(&extW[0]);
		PrecValueType* p_B    = thrust::raw_pointer_cast(&m_B[0]);

		dim3 gridsSweep(numPart_eff, m_k);
		dim3 gridsCopy(m_k, numPart_eff);

		int* p_ks                 = thrust::raw_pointer_cast(&m_ks[1]);
		int* p_offDiagWidths_left = thrust::raw_pointer_cast(&m_offDiagWidths_left[0]);
		int* p_offDiagPerms_left  = thrust::raw_pointer_cast(&m_offDiagPerms_left[0]);

		IntVector tmp_offsets;
		IntVector tmp_secondReordering(m_n, first_partition_size);
		IntVector tmp_secondPerm(m_n, first_partition_size);

		cusp::blas::axpby(m_secondReordering, tmp_secondReordering, tmp_secondReordering, 1.0, -1.0);
		cusp::blas::axpby(m_secondPerm, tmp_secondPerm, tmp_secondPerm, 1.0, -1.0);

		int* p_secondReordering = thrust::raw_pointer_cast(&tmp_secondReordering[first_partition_size]);
		int* p_secondPerm       = thrust::raw_pointer_cast(&tmp_secondPerm[first_partition_size]);

		{
			IntVectorH tmp_offsets_host = m_BOffsets;
			for (int i = m_numPartitions-1; i >= 1; i--)
				tmp_offsets_host[i] -= tmp_offsets_host[1];
			tmp_offsets = tmp_offsets_host;
		}

		int* p_offsets = thrust::raw_pointer_cast(&tmp_offsets[1]);

		int permuteBlockX = n_eff;
		int permuteGridX = 1;
		kernelConfigAdjust(permuteBlockX, permuteGridX, BLOCK_SIZE);
		dim3 gridsPermute(permuteGridX, m_k);

		{
			device::copyWVFromOrToExtendedW_general<PrecValueType><<<gridsCopy, numThreadsToUse>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, false);
			buffer.resize((m_k - (BUF_FACTOR-1) * (m_k / BUF_FACTOR)) * n_eff);
			PrecValueType* p_buffer = thrust::raw_pointer_cast(&buffer[0]);

			for (int i = 0; i < BUF_FACTOR; i++) {
				gridsPermute.y = m_k / BUF_FACTOR;
				if (i == BUF_FACTOR - 1)
					gridsPermute.y = m_k - (BUF_FACTOR-1) * (m_k/BUF_FACTOR);
				device::permute<PrecValueType><<<gridsPermute, permuteBlockX>>>(n_eff, p_extW+i*(m_k/BUF_FACTOR)*n_eff, p_buffer, p_secondPerm);
				thrust::copy(buffer.begin(), buffer.begin()+(gridsPermute.y * n_eff), extW.begin()+i*(m_k/BUF_FACTOR)*n_eff);
			}

			{
				int last_row = 0;
				int first_row = 0;
				for (int i = 0; i < numPart_eff; i++) {
					if (i < remainder)
						last_row += partSize + 1;
					else 
						last_row += partSize;
					device::var::fwdElim_leftSpike_per_partition<PrecValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>> (n_eff, m_ks_host[i+1], m_k - m_offDiagWidths_left_host[i], m_BOffsets_host[i+1]+m_ks_host[i+1], p_B, p_extW, first_row, last_row);
					
					int blockX = last_row - first_row;
					int gridX = 1;
					kernelConfigAdjust(blockX, gridX, BLOCK_SIZE);
					dim3 grids(gridX, m_offDiagWidths_left_host[i]);

					device::var::preBck_leftSpike_divide_per_partition<PrecValueType><<<grids, blockX>>> (n_eff, m_ks_host[i+1], m_k - m_offDiagWidths_left_host[i], m_BOffsets_host[i+1]+m_ks_host[i+1], p_B, p_extW, first_row, last_row);
					device::var::bckElim_leftSpike_per_partition<PrecValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>>(n_eff, m_ks_host[i+1], m_k - m_offDiagWidths_left_host[i], m_BOffsets_host[i+1] + m_ks_host[i+1] + (2*m_ks_host[i+1]+1)*(last_row-first_row-1), p_B, p_extW, first_row, last_row);
					first_row = last_row;
				}
			}

			for (int i = 0; i < BUF_FACTOR; i++) {
				gridsPermute.y = m_k / BUF_FACTOR;
				if (i == BUF_FACTOR - 1)
					gridsPermute.y = m_k - (BUF_FACTOR-1) * (m_k/BUF_FACTOR);

				device::permute<PrecValueType><<<gridsPermute, permuteBlockX>>>(n_eff, p_extW+i*(m_k/BUF_FACTOR)*n_eff, p_buffer, p_secondReordering);
				thrust::copy(buffer.begin(), buffer.begin()+(gridsPermute.y * n_eff), extW.begin()+i*(m_k/BUF_FACTOR)*n_eff);
			}

			device::copyWVFromOrToExtendedW_general<PrecValueType><<<gridsCopy, numThreadsToUse>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, true);
		}

		for (int i = 0; i < numPart_eff; i++) {
			cusp::blas::fill(WV_spare, (PrecValueType) 0);
			device::matrixWReordering_perPartition<PrecValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>>(m_k, p_WV+(2*i+1)*m_k*m_k, p_WV_spare, p_offDiagPerms_left+i*m_k);
			thrust::copy(WV_spare.begin(), WV_spare.end(), WV.begin() + ((2*i+1)*m_k*m_k));
		}
	}
}

template <typename PrecVector>
void
Precond<PrecVector>::calculateSpikes_const(PrecVector&  WV)
{
	PrecValueType* p_WV = thrust::raw_pointer_cast(&WV[0]);


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

		PrecVector extV(m_k * n_eff, (PrecValueType) 0);

		PrecValueType* p_extV = thrust::raw_pointer_cast(&extV[0]);
		PrecValueType* p_B    = thrust::raw_pointer_cast(&m_B[0]);

		dim3 gridsCopy(m_k, numPart_eff);
		dim3 gridsSweep(numPart_eff, m_k);

		if (m_k > 1024) {
			device::copyWVFromOrToExtendedV_general<PrecValueType><<<gridsCopy, 512>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, false);
			device::forwardElimL_bottom_general<PrecValueType><<<gridsSweep, 512>>>(n_eff, m_k, m_k, p_B, p_extV, partSize, remainder);
			device::backwardElimU_bottom_general<PrecValueType><<<gridsSweep, 512>>>(n_eff, m_k, 2*m_k, p_B, p_extV, partSize, remainder);
			device::copyWVFromOrToExtendedV_general<PrecValueType><<<gridsCopy, 512>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, true);
		} else if (m_k > 32) {
			device::copyWVFromOrToExtendedV<PrecValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, false);
			device::forwardElimL_bottom_g32<PrecValueType><<<gridsSweep, m_k>>>(n_eff, m_k, m_k, p_B, p_extV, partSize, remainder);
			device::backwardElimU_bottom_g32<PrecValueType><<<gridsSweep, m_k>>>(n_eff, m_k, 2*m_k, p_B, p_extV, partSize, remainder);
			device::copyWVFromOrToExtendedV<PrecValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, true);
		} else {
			device::copyWVFromOrToExtendedV<PrecValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, false);
			device::forwardElimL_bottom<PrecValueType><<<gridsSweep, m_k>>>(n_eff, m_k, m_k, p_B, p_extV, partSize, remainder);
			device::backwardElimU_bottom<PrecValueType><<<gridsSweep, m_k>>>(n_eff, m_k, 2*m_k, p_B, p_extV, partSize, remainder);
			device::copyWVFromOrToExtendedV<PrecValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extV, true);
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

		PrecVector  extW(m_k * n_eff, (PrecValueType) 0);

		PrecValueType* p_extW = thrust::raw_pointer_cast(&extW[0]);
		PrecValueType* p_B    = thrust::raw_pointer_cast(&m_B[(2*m_k+1)*first_partition_size]);

		dim3 gridsSweep(numPart_eff, m_k);
		dim3 gridsCopy(m_k, numPart_eff);

		if (m_k > 1024) {
			device::copyWVFromOrToExtendedW_general<PrecValueType><<<gridsCopy, 512>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, false);
			device::forwardElimL_general<PrecValueType><<<gridsSweep, 512>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::backwardElimU_general<PrecValueType><<<gridsSweep, 512>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::copyWVFromOrToExtendedW_general<PrecValueType><<<gridsCopy, 512>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, true);
		} else if (m_k > 32) {
			device::copyWVFromOrToExtendedW<PrecValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, false);
			device::forwardElimL_g32<PrecValueType><<<gridsSweep, m_k>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::backwardElimU_g32<PrecValueType><<<gridsSweep, m_k>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::copyWVFromOrToExtendedW<PrecValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, true);
		} else {
			device::copyWVFromOrToExtendedW<PrecValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, false);
			device::forwardElimL<PrecValueType><<<gridsSweep, m_k>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::backwardElimU<PrecValueType><<<gridsSweep, m_k>>>(n_eff, m_k, p_B, p_extW, partSize, remainder);
			device::copyWVFromOrToExtendedW<PrecValueType><<<gridsCopy, m_k>>>(n_eff, m_k, partSize, remainder, p_WV, p_extW, true);
		}
	}
}

template <typename PrecVector>
void
Precond<PrecVector>::calculateSpikes_var(PrecVector&  WV)
{
	PrecVector WV_spare(m_k*m_k);

	PrecValueType* p_WV       = thrust::raw_pointer_cast(&WV[0]);
	PrecValueType* p_WV_spare = thrust::raw_pointer_cast(&WV_spare[0]);

	// Calculate the size of the first and last partitions.
	int numThreadsToUse = adjustNumThreads(cusp::blas::nrm1(m_ks) / m_numPartitions);

	const int SWEEP_MAX_NUM_THREADS = 128;
	// Copy WV into extV, perform sweeps to calculate extV, then copy back extV to WV.
	// Note that we skip the last partition (no right spike associated with it). Also
	// note that we only perform truncated spikes using the bottom parts of the L and
	// U factors to calculate the bottom block of the right spikes V.
	{
		int n_eff             = m_n;
		int numPart_eff       = m_numPartitions;
		int partSize          = n_eff / numPart_eff;
		int remainder         = n_eff % numPart_eff;
		int rightOffDiagWidth = cusp::blas::nrmmax(m_offDiagWidths_right);
		int leftOffDiagWidth  = cusp::blas::nrmmax(m_offDiagWidths_left);

		PrecVector extWV((leftOffDiagWidth + rightOffDiagWidth) * n_eff, (PrecValueType) 0);
		PrecVector buffer;

		PrecValueType* p_extWV               = thrust::raw_pointer_cast(&extWV[0]);
		PrecValueType* p_B                   = thrust::raw_pointer_cast(&m_B[0]);
		int*           p_secondReordering    = thrust::raw_pointer_cast(&m_secondReordering[0]);
		int*           p_secondPerm          = thrust::raw_pointer_cast(&m_secondPerm[0]);
		int*           p_ks                  = thrust::raw_pointer_cast(&m_ks[0]);
		int*           p_offDiagWidths_right = thrust::raw_pointer_cast(&m_offDiagWidths_right[0]);
		int*           p_offDiagPerms_right  = thrust::raw_pointer_cast(&m_offDiagPerms_right[0]);
		int*           p_offDiagWidths_left  = thrust::raw_pointer_cast(&m_offDiagWidths_left[0]);
		int*           p_offDiagPerms_left   = thrust::raw_pointer_cast(&m_offDiagPerms_left[0]);
		int*           p_first_rows          = thrust::raw_pointer_cast(&m_first_rows[0]);
		int*           p_offsets             = thrust::raw_pointer_cast(&m_BOffsets[0]);

		int permuteBlockX = leftOffDiagWidth+rightOffDiagWidth;
		int permuteGridX  = 1;
		int permuteGridY  = n_eff;
		int permuteGridZ  = 1;
		kernelConfigAdjust(permuteBlockX, permuteGridX, BLOCK_SIZE);
		kernelConfigAdjust(permuteGridY, permuteGridZ, MAX_GRID_DIMENSION);
		dim3 gridsPermute(permuteGridX, permuteGridY, permuteGridZ);

		buffer.resize((leftOffDiagWidth + rightOffDiagWidth) * n_eff);
		
		PrecValueType* p_buffer = thrust::raw_pointer_cast(&buffer[0]);

		dim3 gridsCopy((leftOffDiagWidth + rightOffDiagWidth), numPart_eff);

		device::copyWVFromOrToExtendedWVTranspose_general<PrecValueType><<<gridsCopy, numThreadsToUse>>>(leftOffDiagWidth + rightOffDiagWidth, m_k, rightOffDiagWidth, partSize, remainder, m_k-rightOffDiagWidth-leftOffDiagWidth, p_WV, p_extWV, false);
		device::columnPermute<PrecValueType><<<gridsPermute, permuteBlockX>>>(n_eff, leftOffDiagWidth+rightOffDiagWidth, p_extWV, p_buffer, p_secondPerm);

		{				
			int sweepBlockX = leftOffDiagWidth;
			int sweepGridX = 1;
			if (sweepBlockX < rightOffDiagWidth)
				sweepBlockX = rightOffDiagWidth;
			kernelConfigAdjust(sweepBlockX, sweepGridX, SWEEP_MAX_NUM_THREADS);
			dim3 sweepGrids(sweepGridX, 2*numPart_eff-2);

			device::var::fwdElim_spike<PrecValueType><<<sweepGrids, sweepBlockX>>>(n_eff, p_ks, leftOffDiagWidth + rightOffDiagWidth, rightOffDiagWidth, p_offsets, p_B, p_buffer, partSize, remainder, p_offDiagWidths_left, p_offDiagWidths_right, p_first_rows);

			int preBckBlockX = leftOffDiagWidth + rightOffDiagWidth;
			int preBckGridX  = 1;
			int preBckGridY  = n_eff;
			int preBckGridZ  = 1;
			kernelConfigAdjust(preBckBlockX, preBckGridX, BLOCK_SIZE);
			kernelConfigAdjust(preBckGridY, preBckGridZ, MAX_GRID_DIMENSION);
			dim3 preBckGrids(preBckGridX, preBckGridY, preBckGridZ);

			device::var::preBck_offDiag_divide<PrecValueType><<<preBckGrids, preBckBlockX>>>(n_eff, leftOffDiagWidth + rightOffDiagWidth, p_ks, p_offsets, p_B, p_buffer, partSize, remainder);

			{
				int last_row = 0;
				for (int i = 0; i < m_numPartitions - 1; i++) {
					if (i < remainder)
						last_row += (partSize + 1);
					else
						last_row += partSize;

					m_first_rows_host[i] = thrust::reduce(m_secondPerm.begin()+(last_row-m_k), m_secondPerm.begin()+last_row, last_row, thrust::minimum<int>());
				}
				m_first_rows = m_first_rows_host;
				p_first_rows = thrust::raw_pointer_cast(&m_first_rows[0]);
			}

			device::var::bckElim_spike<PrecValueType><<<sweepGrids, sweepBlockX>>>(n_eff, p_ks, leftOffDiagWidth + rightOffDiagWidth, rightOffDiagWidth, p_offsets, p_B, p_buffer, partSize, remainder, p_offDiagWidths_left, p_offDiagWidths_right, p_first_rows);
		}


		device::columnPermute<PrecValueType><<<gridsPermute, permuteBlockX>>>(n_eff, leftOffDiagWidth + rightOffDiagWidth, p_buffer, p_extWV, p_secondReordering);
		device::copyWVFromOrToExtendedWVTranspose_general<PrecValueType><<<gridsCopy, numThreadsToUse>>>(leftOffDiagWidth + rightOffDiagWidth, m_k, rightOffDiagWidth, partSize, remainder, m_k-rightOffDiagWidth-leftOffDiagWidth, p_WV, p_extWV, true);

		for (int i = 0; i < numPart_eff - 1; i++) {
			cusp::blas::fill(WV_spare, (PrecValueType) 0);
			device::matrixVReordering_perPartition<PrecValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>>(m_k, p_WV+2*i*m_k*m_k, p_WV_spare, p_offDiagPerms_right+i*m_k);
			thrust::copy(WV_spare.begin(), WV_spare.end(), WV.begin()+(2*i*m_k*m_k));

			cusp::blas::fill(WV_spare, (PrecValueType) 0);
			device::matrixWReordering_perPartition<PrecValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>>(m_k, p_WV+(2*i+1)*m_k*m_k, p_WV_spare, p_offDiagPerms_left+i*m_k);
			thrust::copy(WV_spare.begin(), WV_spare.end(), WV.begin()+((2*i+1)*m_k*m_k));
		}
	}
}

/**
 * This function adjust the number of threads used for kernels which can take
 * any number of threads.
 */
template <typename PrecVector>
int
Precond<PrecVector>::adjustNumThreads(int inNumThreads) {
	int prev = 0;
	int cur;
	
	for (int i = 0; i < 16; i++) {
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

/**
 * This function calculates the spike blocks in the LU_UL case.
 */
template <typename PrecVector>
void
Precond<PrecVector>::calculateSpikes(PrecVector&  B2,
                                     PrecVector&  WV)
{
	int  two_k     = 2 * m_k;
	int  partSize  = m_n / m_numPartitions;
	int  remainder = m_n % m_numPartitions;

	// Compress the provided UL factorization 'B2' into 'compB2'.
	PrecVector compB2((two_k+1)*two_k*(m_numPartitions-1));
	cusp::blas::fill(compB2, (PrecValueType) 0);

	PrecValueType* p_B2     = thrust::raw_pointer_cast(&B2[0]);
	PrecValueType* p_compB2 = thrust::raw_pointer_cast(&compB2[0]);

	dim3 gridsCompress(two_k, m_numPartitions-1);

	if (m_k > 511)
		device::copydAtodA2_general<PrecValueType><<<gridsCompress, 1024>>>(m_n, m_k, p_B2, p_compB2, two_k, partSize, m_numPartitions, remainder);
	else
		device::copydAtodA2<PrecValueType><<<gridsCompress, two_k+1>>>(m_n, m_k, p_B2, p_compB2, two_k, partSize, m_numPartitions, remainder);

	// Combine 'B' and 'compB2' into 'partialB'.
	PrecVector partialB(2*(two_k+1)*(m_k+1)*(m_numPartitions-1));

	PrecValueType* p_B        = thrust::raw_pointer_cast(&m_B[0]);
	PrecValueType* p_partialB = thrust::raw_pointer_cast(&partialB[0]);

	dim3 gridsCopy(m_k+1, 2*(m_numPartitions-1));

	if (m_k > 511)
		device::copydAtoPartialA_general<PrecValueType><<<gridsCopy, 1024>>>(m_n, m_k, p_B, p_compB2, p_partialB, partSize, m_numPartitions, remainder, two_k);
	else
		device::copydAtoPartialA<PrecValueType><<<gridsCopy, two_k+1>>>(m_n, m_k, p_B, p_compB2, p_partialB, partSize, m_numPartitions, remainder, two_k);

	// Perform forward/backward sweeps to calculate the spike blocks 'W' and 'V'.
	PrecValueType* p_WV = thrust::raw_pointer_cast(&WV[0]);

	dim3 gridsSweep(m_numPartitions-1, m_k);

	if (m_k > 1024) {
		device::forwardElimLdWV_general<PrecValueType><<<gridsSweep, 512>>>(m_k, p_partialB, p_WV, m_k, 0, 0);
		device::backwardElimUdWV_general<PrecValueType><<<gridsSweep, 512>>>(m_k, p_partialB, p_WV, m_k, 0, 1);
		device::backwardElimUdWV_general<PrecValueType><<<gridsSweep, 512>>>(m_k, p_partialB, p_WV, m_k, 1, 0);
		device::forwardElimLdWV_general<PrecValueType><<<gridsSweep, 512>>>(m_k, p_partialB, p_WV, m_k, 1, 1);
	} else if (m_k > 32)  {
		device::forwardElimLdWV_g32<PrecValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 0, 0);
		device::backwardElimUdWV_g32<PrecValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 0, 1);
		device::backwardElimUdWV_g32<PrecValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 1, 0);
		device::forwardElimLdWV_g32<PrecValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 1, 1);
	} else {
		device::forwardElimLdWV<PrecValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 0, 0);
		device::backwardElimUdWV<PrecValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 0, 1);
		device::backwardElimUdWV<PrecValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 1, 0);
		device::forwardElimLdWV<PrecValueType><<<gridsSweep, m_k>>>(m_k, p_partialB, p_WV, m_k, 1, 1);
	}
}

/**
 * This function assembles the truncated Spike reduced matrix R.
 */
template <typename PrecVector>
void
Precond<PrecVector>::assembleReducedMat(PrecVector&  WV)
{
	PrecValueType* p_WV = thrust::raw_pointer_cast(&WV[0]);
	PrecValueType* p_R  = thrust::raw_pointer_cast(&m_R[0]);

	dim3 grids(m_k, m_numPartitions-1);

	if (!m_variableBandwidth) {
		if (m_k > 1024)
			device::assembleReducedMat_general<PrecValueType><<<grids, 512>>>(m_k, p_WV, p_R);
		else if (m_k > 32)
			device::assembleReducedMat_g32<PrecValueType><<<grids, m_k>>>(m_k, p_WV, p_R);
		else
			device::assembleReducedMat<PrecValueType><<<m_numPartitions-1, m_k*m_k>>>(m_k, p_WV, p_R);
	} else {
		int* p_WVOffsets = thrust::raw_pointer_cast(&m_WVOffsets[0]);
		int* p_ROffsets  = thrust::raw_pointer_cast(&m_ROffsets[0]);
		int* p_spike_ks  = thrust::raw_pointer_cast(&m_spike_ks[0]);
	
		if (m_k > 1024)
			device::assembleReducedMat_var_bandwidth_general<PrecValueType><<<grids, 512>>>(p_spike_ks, p_WVOffsets, p_ROffsets, p_WV, p_R);
		else if (m_k > 32)
			device::assembleReducedMat_var_bandwidth_g32<PrecValueType><<<grids, m_k>>>(p_spike_ks, p_WVOffsets, p_ROffsets, p_WV, p_R);
		else
			device::assembleReducedMat_var_bandwidth<PrecValueType><<<m_numPartitions-1, m_k*m_k>>>(p_spike_ks, p_WVOffsets, p_ROffsets, p_WV, p_R);
	}
}

/**
 * This function copies the last partition from B2, which contains the UL results,
 * to m_B.
 */
template <typename PrecVector>
void
Precond<PrecVector>::copyLastPartition(PrecVector &B2) {
	thrust::copy(B2.begin()+(2*m_k+1) * (m_n - m_n / m_numPartitions), B2.end(), m_B.begin()+(2*m_k+1) * (m_n - m_n / m_numPartitions) );
}


/**
 * This function checks the diagonal of the specified banded matrix for any 
 * elements that are smaller in absolute value than a threshold value
 */
template <typename T>
struct zero_functor : thrust::unary_function<T, bool> 
{
	zero_functor(T threshold) : m_threshold(threshold) {}

	__host__ __device__
	bool operator()(T val) {return abs(val) < m_threshold;}

	T  m_threshold;
};


template <typename PrecVector>
bool
Precond<PrecVector>::hasZeroPivots(const PrecVectorIterator&    start_B,
                                   const PrecVectorIterator&    end_B,
                                   int                          k,
                                   PrecValueType                threshold)
{
	// Create a strided range to select the main diagonal
	strided_range<typename PrecVector::iterator> diag(start_B + k, end_B, 2*k + 1);

	////std::cout << std::endl;
	////thrust::copy(diag.begin(), diag.end(), std::ostream_iterator<PrecValueType>(std::cout, " "));
	////std::cout << std::endl;

	// Check if any of the diagonal elements is within the specified threshold
	return thrust::any_of(diag.begin(), diag.end(), zero_functor<PrecValueType>(threshold));
}



} // namespace spike


#endif
