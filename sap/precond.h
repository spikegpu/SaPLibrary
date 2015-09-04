/** \file precond.h
 *  \brief Definition of the SaP preconditioner class.
 */

#ifndef SAP_PRECOND_CUH
#define SAP_PRECOND_CUH

#include <sap/common.h>
#include <sap/graph.h>
#include <sap/timer.h>
#include <sap/strided_range.h>
#include <sap/device/factor_band_const.cuh>
#include <sap/device/factor_band_var.cuh>
#include <sap/device/sweep_band_const.cuh>
#include <sap/device/sweep_band_var.cuh>
#include <sap/device/sweep_band_sparse.cuh>
#include <sap/device/inner_product.cuh>
#include <sap/device/shuffle.cuh>
#include <sap/device/data_transfer.cuh>

#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#include <cusp/detail/format_utils.h>
#else
#include <cusp/blas/blas.h>
#include <cusp/format_utils.h>
#endif

#include <cusp/print.h>

#include <thrust/logical.h>
#include <thrust/functional.h>

#include <omp.h>
#include <queue>
#include <vector>
#include <functional>
#include <stdlib.h>

namespace sap {

/// SaP preconditioner.
/**
 * This class implements the SaP preconditioner.
 *
 * \tparam PrecVector is the vector type used in the preconditioner.
 *         (its underlying type defines the precision of the preconditioner).
 */
template <typename PrecVector>
class Precond
{
public:
    typedef typename PrecVector::memory_space  MemorySpace;
    typedef typename PrecVector::memory_space  memory_space;
    typedef typename PrecVector::value_type    PrecValueType;
    typedef typename PrecVector::value_type    value_type;
    typedef typename PrecVector::iterator      PrecVectorIterator;
    typedef typename cusp::unknown_format      format;

    typedef typename cusp::array1d<int, MemorySpace>                  IntVector;
    typedef IntVector                                                 MatrixMap;
    typedef typename cusp::array1d<PrecValueType, MemorySpace>        MatrixMapF;

    typedef typename cusp::array1d<PrecValueType, cusp::host_memory>  PrecVectorH;
    typedef typename cusp::array1d<int, cusp::host_memory>            IntVectorH;
    typedef typename cusp::array1d<bool, cusp::host_memory>           BoolVectorH;
    typedef IntVectorH                                                MatrixMapH;
    typedef typename cusp::array1d<PrecValueType, cusp::host_memory>  MatrixMapFH;

    typedef typename IntVectorH::iterator                             IntHIterator;
    typedef typename PrecVectorH::iterator                            PrecHIterator;

    typedef typename cusp::coo_matrix<int, PrecValueType, MemorySpace>        PrecMatrixCoo;
    typedef typename cusp::coo_matrix<int, PrecValueType, cusp::host_memory>  PrecMatrixCooH;
    typedef typename cusp::csr_matrix<int, PrecValueType, MemorySpace>        PrecMatrixCsr;
    typedef typename cusp::csr_matrix<int, PrecValueType, cusp::host_memory>  PrecMatrixCsrH;


    Precond();

    Precond(int                 numPart,
            bool                isSPD,
            bool                saveMem,
            bool                reorder,
            bool                testDB,
            bool                doDB,
            bool                dbFirstStageOnly,
            bool                scale,
            double              dropOff_frac,
            int                 maxBandwidth,
            int                 gpuCount,
            FactorizationMethod factMethod,
            PreconditionerType  precondType,
            bool                safeFactorization,
            bool                variableBandwidth,
            bool                trackReordering,
            int                 ilu_level,
            PrecValueType       tolerance);

    Precond(const Precond&  prec);

    ~Precond() {}

    Precond & operator = (const Precond &prec);

    double getTimeDB() const            {return m_time_DB;}
    double getTimeDBPre() const         {return m_time_DB_pre;}
    double getTimeDBFirst() const       {return m_time_DB_first;}
    double getTimeDBSecond() const      {return m_time_DB_second;}
    double getTimeDBPost() const        {return m_time_DB_post;}
    double getTimeReorder() const         {return m_time_reorder;}
    double getTimeDropOff() const         {return m_time_dropOff;}
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
    int    getBandwidthDB() const       {return m_k_db;}
    int    getBandwidth() const           {return m_k;}

    int    getNumPartitions() const       {return m_numPartitions;}
    double getActualDropOff() const       {return (double) m_dropOff_actual;}

    int    getActualNumNonZeros() const   {return m_actual_nnz;}

    //// NOTE:  Matrix here will usually be PrecMatrixCooH, except
    ////        when there's a single component when it will be whatever
    ////        the user passes to Solver::setup().
    template <typename Matrix>
    void   setup(const Matrix&  A);

    void   update(const PrecVector& entries);

    void   solve(PrecVector& v, PrecVector& z);

    template <typename SolverVector>
    void   operator()(const SolverVector& v, SolverVector& z);

private:
    int                  m_numPartitions;
    int                  m_n;
    int                  m_k;

    bool                 m_isSPD;
    bool                 m_saveMem; 
    bool                 m_reorder;
    bool                 m_testDB;
    bool                 m_doDB;
    bool                 m_dbFirstStageOnly;
    bool                 m_scale;
    PrecValueType        m_dropOff_frac;
    int                  m_maxBandwidth;
    int                  m_gpuCount;
    FactorizationMethod  m_factMethod;
    PreconditionerType   m_precondType;
    bool                 m_safeFactorization;
    bool                 m_variableBandwidth;
    bool                 m_trackReordering;

    int                  m_ilu_level;
    PrecValueType        m_tolerance;

    MatrixMap            m_offDiagMap;
    MatrixMap            m_WVMap;
    MatrixMap            m_typeMap;
    MatrixMap            m_bandedMatMap;
    MatrixMapF           m_scaleMap;

    // Used in variable-bandwidth method only, host versions
    IntVectorH           m_ks_host;

public:
    IntVectorH           m_ks_row_host;
    IntVectorH           m_ks_col_host;

private:
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

    PrecVector           m_dbRowScale;            // DB row scaling
    PrecVector           m_dbColScale;            // DB col scaling

    PrecVector           m_buffer;
    PrecVector           m_buffer2;

    std::vector<PrecVector>   m_all_Bs;               // For multi-GPU only
    std::vector<IntVector>    m_all_BOffsets;         // For multi-GPU only
    std::vector<IntVector>    m_all_ks;               // For multi-GPU only
    std::vector<IntVectorH>   m_all_BOffsets_host;    // For multi-GPU only
    std::vector<IntVectorH>   m_all_ks_host;          // For multi-GPU only
    IntVectorH           m_all_num_rows;              // For multi-GPU only
    IntVectorH           m_all_num_partitions;        // For multi-GPU only

    PrecVector           m_B;                     // banded matrix (LU factors)
    PrecVector           m_B2;                    // banded matrix (LU factors)
    PrecVectorH          m_Bh;                    // FIXME: currently for hacking purpose only, to be removed
    PrecVector           m_offDiags;              // contains the off-diagonal blocks of the original banded matrix
    PrecVector           m_R;                     // diagonal blocks in the reduced matrix (LU factors)
    PrecMatrixCsrH       m_Acsrh;
    PrecMatrixCsrH       m_Acsrh_ul;
    PrecVectorH          m_pivots;
    PrecVectorH          m_pivots_ul;

    PrecVectorH          m_offDiags_host;         // Used with second-stage reorder only, copy the offDiags in SpikeGragh
    PrecVectorH          m_WV_host;

    int                  m_k_reorder;             // bandwidth after reordering
    int                  m_k_db;                  // bandwidth after DB

    int                  m_actual_nnz;            // The actual number of non-zeros after LU

    PrecValueType        m_dropOff_actual;        // actual dropOff fraction achieved

    // Temporary vectors used in preconditioner solve (to support mixed-precision).
    PrecVector           m_vp;                    // copy of specified RHS vector
    PrecVector           m_zp;                    // copy of solution vector

    GPUTimer             m_timer;
    double               m_time_DB;               // CPU time for DB reordering
    double               m_time_DB_pre;           // CPU time for DB reordering (pre-processing)
    double               m_time_DB_first;         // CPU time for DB reordering (first stage)
    double               m_time_DB_second;        // CPU time for DB reordering (second stage)
    double               m_time_DB_post;          // CPU time for DB reordering (post-processing)
    double               m_time_reorder;          // CPU time for DB matrix reordering
    double               m_time_dropOff;          // CPU time for drop off
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
    void partBlockedBandedLU_one();
    void partBlockedBandedLU_const();
    void partBlockedBandedCholesky_one();
    void partBlockedBandedCholesky_var();
    void partBandedLU_var();
    void partBlockedBandedLU_var();
    void partBandedUL(PrecVector& B);
    void partBlockedBandedUL(PrecVector& B);
    void sparseFactorization();

    void partBandedFwdSweep(PrecVector& v);
    void partBandedFwdSweep_const(PrecVector& v);
    void partBandedFwdSweep_var(PrecVector& v);
    void partBandedBckSweep(PrecVector& v);
    void partBandedBckSweep_const(PrecVector& v);
    void partBandedBckSweep_var(PrecVector& v);
    void partBandedSweepsH(PrecVector& v);
    void sparseSweep(PrecVector& v, PrecVector& w);

    void partFullLU();
    void partFullLU_const();
    void partFullLU_var();
    void partBlockedFullLU_var();

    void ILU0(PrecMatrixCsrH& Acsrh);
    void ILUT(PrecMatrixCsrH& Acsrh, int p, PrecValueType tau);
    void ILUULT(PrecMatrixCsrH& Acsrh, PrecMatrixCsrH& Acsrh2, int p, PrecValueType tau);
    void ILUTP(PrecMatrixCsrH&    Acsrh,
               int                p,
               PrecValueType      tau,
               PrecValueType      perm_tol,
               IntVectorH&        perm,
               IntVectorH&        reordering);

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
                       int                       step,
                       PrecValueType             threshold);

    void findPthMax(const IntHIterator&          ibegin,
                    const IntHIterator&          iend,
                    const PrecHIterator&         vbegin,
                    const PrecHIterator&         vend,
                    int                          p);
};


// Functor objects 
// 
// TODO:  figure out why I cannot make these private to Precond...
template<typename T>
struct Multiply: public thrust::unary_function<T, T>
{
    __host__ __device__
    T operator() (thrust::tuple<T, T> tu) {
        return thrust::get<0>(tu) * thrust::get<1>(tu);
    }
};

template <typename T>
struct SmallerThan : public thrust::unary_function<T, bool> 
{
    SmallerThan(T threshold) : m_threshold(threshold) {}

    __host__ __device__
    bool operator()(T val) {return std::abs(val) < m_threshold;}

    T  m_threshold;
};


/**
 * This is the constructor for the Precond class.
 */
template <typename PrecVector>
Precond<PrecVector>::Precond(int                 numPart,
                             bool                isSPD,
                             bool                saveMem,
                             bool                reorder,
                             bool                testDB,
                             bool                doDB,
                             bool                dbFirstStageOnly,
                             bool                scale,
                             double              dropOff_frac,
                             int                 maxBandwidth,
                             int                 gpuCount,
                             FactorizationMethod factMethod,
                             PreconditionerType  precondType,
                             bool                safeFactorization,
                             bool                variableBandwidth,
                             bool                trackReordering,
                             int                 ilu_level,
                             PrecValueType       tolerance)
:   m_numPartitions(numPart),
    m_isSPD(isSPD),
    m_saveMem(saveMem),
    m_reorder(reorder),
    m_testDB(testDB),
    m_doDB(doDB),
    m_dbFirstStageOnly(dbFirstStageOnly),
    m_scale(scale),
    m_dropOff_frac((PrecValueType)dropOff_frac),
    m_maxBandwidth(maxBandwidth),
    m_gpuCount(gpuCount),
    m_factMethod(factMethod),
    m_precondType(precondType),
    m_safeFactorization(safeFactorization),
    m_variableBandwidth(variableBandwidth),
    m_trackReordering(trackReordering),
    m_ilu_level(ilu_level),
    m_tolerance(tolerance),
    m_k_reorder(0),
    m_k_db(0),
    m_k(0),
    m_actual_nnz(0),
    m_dropOff_actual(0),
    m_time_reorder(0),
    m_time_DB(0),
    m_time_DB_pre(0),
    m_time_DB_first(0),
    m_time_DB_second(0),
    m_time_DB_post(0),
    m_time_dropOff(0),
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
:   m_isSPD(false),
    m_saveMem(false),
    m_reorder(false),
    m_testDB(false),
    m_doDB(false),
    m_dbFirstStageOnly(false),
    m_scale(false),
    m_k_reorder(0),
    m_k_db(0),
    m_k(0),
    m_actual_nnz(0),
    m_dropOff_actual(0),
    m_maxBandwidth(std::numeric_limits<int>::max()),
    m_gpuCount(1),
    m_time_reorder(0),
    m_time_DB(0),
    m_time_DB_pre(0),
    m_time_DB_first(0),
    m_time_DB_second(0),
    m_time_DB_post(0),
    m_time_dropOff(0),
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
:   m_k_reorder(0),
    m_k_db(0),
    m_k(0),
    m_dropOff_actual(0),
    m_time_reorder(0),
    m_time_DB(0),
    m_time_DB_pre(0),
    m_time_DB_first(0),
    m_time_DB_second(0),
    m_time_DB_post(0),
    m_time_dropOff(0),
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
    m_numPartitions      = prec.m_numPartitions;

    m_isSPD              = prec.m_isSPD;
    m_saveMem            = prec.m_saveMem;
    m_reorder            = prec.m_reorder;
    m_testDB             = prec.m_testDB;
    m_doDB               = prec.m_doDB;
    m_dbFirstStageOnly   = prec.m_dbFirstStageOnly;
    m_scale              = prec.m_scale;
    m_dropOff_frac       = prec.m_dropOff_frac;
    m_maxBandwidth       = prec.m_maxBandwidth;
    m_gpuCount           = prec.m_gpuCount;
    m_factMethod         = prec.m_factMethod;
    m_precondType        = prec.m_precondType;
    m_safeFactorization  = prec.m_safeFactorization;
    m_variableBandwidth  = prec.m_variableBandwidth;
    m_trackReordering    = prec.m_trackReordering;
    m_ilu_level          = prec.m_ilu_level;
    m_tolerance          = prec.m_tolerance;
    m_actual_nnz         = prec.m_actual_nnz;
}

template <typename PrecVector>
Precond<PrecVector>& 
Precond<PrecVector>::operator=(const Precond<PrecVector>& prec)
{
    m_numPartitions      = prec.m_numPartitions;

    m_isSPD              = prec.m_isSPD;
    m_saveMem            = prec.m_saveMem;
    m_reorder            = prec.m_reorder;
    m_testDB             = prec.m_testDB;
    m_doDB               = prec.m_doDB;
    m_dbFirstStageOnly   = prec.m_dbFirstStageOnly;
    m_scale              = prec.m_scale;
    m_dropOff_frac       = prec.m_dropOff_frac;
    m_maxBandwidth       = prec.m_maxBandwidth;
    m_gpuCount           = prec.m_gpuCount;
    m_factMethod         = prec.m_factMethod;
    m_precondType        = prec.m_precondType;
    m_safeFactorization  = prec.m_safeFactorization;
    m_variableBandwidth  = prec.m_variableBandwidth;
    m_trackReordering    = prec.m_trackReordering;
    m_ilu_level          = prec.m_ilu_level;
    m_tolerance          = prec.m_tolerance;
    m_actual_nnz         = prec.m_actual_nnz;

    m_k                        = prec.m_k;
    m_ks_host                  = prec.m_ks_host;
    m_offDiagWidths_left_host  = prec.m_offDiagWidths_left_host;
    m_offDiagWidths_right_host = prec.m_offDiagWidths_right_host;
    m_first_rows_host          = prec.m_first_rows_host;
    m_BOffsets_host            = prec.m_BOffsets_host;

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
    m_time_reorder = 0.0;

    m_timer.Start();


    cusp::blas::fill(m_B, (PrecValueType) 0);

    thrust::scatter_if(
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.begin(), m_scaleMap.begin())), Multiply<PrecValueType>()),
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.end(), m_scaleMap.end())), Multiply<PrecValueType>()),
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
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.begin(), m_scaleMap.begin())), Multiply<PrecValueType>()),
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.end(), m_scaleMap.end())), Multiply<PrecValueType>()),
            m_offDiagMap.begin(),
            m_typeMap.begin(),
            m_offDiags.begin(),
            thrust::logical_not<int>()
            );

    thrust::scatter_if(
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.begin(), m_scaleMap.begin())), Multiply<PrecValueType>()),
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(entries.end(), m_scaleMap.end())), Multiply<PrecValueType>()),
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
            // partBandedUL(B2);
            partBlockedBandedUL(B2);
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
 * (1) Reorder the matrix (DB and/or RCM)
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

    if (m_precondType == None)
        return;

    // Form the banded matrix based on the specified matrix, either through
    // transformation (reordering and drop-off) or straight conversion.
    if (m_reorder)
        transformToBandedMatrix(A);
    else
        convertToBandedMatrix(A);

    // Allocate space for vectors used to interface the Krylov solver to 
    // the preconditioner solve function (while allowing for different types).
    m_vp.resize(m_n);
    m_zp.resize(m_n);
    m_buffer.resize(m_n);
    m_buffer2.resize(m_n);

    // For DB test only, directly exit
    if (m_testDB)
        return;

    ////cusp::io::write_matrix_market_file(m_B, "B.mtx");
    if (m_k == 0) {
        m_actual_nnz = (2 * m_k + 1) * m_n - thrust::count(m_B.begin(), m_B.end(), 0.0);
        return;
    }

    if (m_ilu_level >= 0) {
        m_timer.Start();
        sparseFactorization();
        m_timer.Stop();
        m_time_bandLU = m_timer.getElapsed();

        if (m_precondType == Block || m_numPartitions == 1)
            return;

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

        m_timer.Start();
        calculateSpikes(mat_WV);
        assembleReducedMat(mat_WV);
        m_timer.Stop();
        m_time_assembly = m_timer.getElapsed();

        m_timer.Start();
        partFullLU();
        m_timer.Stop();
        m_time_fullLU = m_timer.getElapsed();
        return;
    }

    // If we are using a single partition, perform the LU factorization
    // of the banded matrix and return.
    if (m_precondType == Block || m_numPartitions == 1) {

        m_timer.Start();
        partBandedLU();
        // m_Bh = m_B;
        m_actual_nnz = (2 * m_k + 1) * m_n - thrust::count(m_B.begin(), m_B.end(), 0.0);
        m_timer.Stop();
        m_time_bandLU = m_timer.getElapsed();
        ////cusp::io::write_matrix_market_file(m_B, "B_lu.mtx");
        return;
    }

    PrecVector mat_WV;

    try {
        // We are using more than one partition, so we must assemble the
        // truncated Spike reduced matrix R.
        m_R.resize((2 * m_k) * (2 * m_k) * (m_numPartitions - 1));

        // Extract off-diagonal blocks from the banded matrix and store them
        // in the array m_offDiags.
        mat_WV.resize(2 * m_k * m_k * (m_numPartitions-1));

        m_timer.Start();
        extractOffDiagonal(mat_WV);
        m_timer.Stop();
        m_time_offDiags = m_timer.getElapsed();
    } catch (const std::bad_alloc& ) {
        m_precondType = Block;
        m_timer.Start();
        partBandedLU();
        // m_Bh = m_B;
        m_actual_nnz = (2 * m_k + 1) * m_n - thrust::count(m_B.begin(), m_B.end(), 0.0);
        m_timer.Stop();
        m_time_bandLU = m_timer.getElapsed();
        return;
    }
    

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
            try{
                m_timer.Start();
                calculateSpikes(mat_WV);
                assembleReducedMat(mat_WV);
                m_timer.Stop();
                m_time_assembly = m_timer.getElapsed();
            } catch (const std::bad_alloc& ) {
                m_precondType = Block;
                m_actual_nnz = (2 * m_k + 1) * m_n - thrust::count(m_B.begin(), m_B.end(), 0.0);
                return;
            }
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
            // partBandedUL(B2);
            partBlockedBandedUL(B2);
            m_timer.Stop();
            m_time_bandUL = m_timer.getElapsed();

            ////cusp::io::write_matrix_market_file(B2, "B_ul.mtx");

            cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);

            try {
                m_timer.Start();
                calculateSpikes(B2, mat_WV);
                assembleReducedMat(mat_WV);
                copyLastPartition(B2);
                {
                    PrecValueType *dB = thrust::raw_pointer_cast(&m_B[0]);
                    int gridX = m_n, gridY = 1;
                    kernelConfigAdjust(gridX, gridY, MAX_GRID_DIMENSION);
                    dim3 grids(gridX, gridY);
                    if (m_k > 512)
                        device::bandLUUL_post_divide_general<PrecValueType><<<grids, 512>>>(dB, m_n, m_k, m_numPartitions);
                    else
                        device::bandLUUL_post_divide<PrecValueType><<<grids, m_k>>>(dB, m_n, m_k, m_numPartitions);
                }
                m_timer.Stop();
                m_time_assembly = m_timer.getElapsed();
            } catch (const std::bad_alloc& ) {
                m_precondType = Block;
                m_actual_nnz = (2 * m_k + 1) * m_n - thrust::count(m_B.begin(), m_B.end(), 0.0);
                return;
            }
        }

        break;
    }
    // m_Bh = m_B;
    m_actual_nnz = (2 * m_k + 1) * m_n - thrust::count(m_B.begin(), m_B.end(), 0.0);

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
 * This is the wrapper around the preconditioner solve function. It is
 * invoked by the Krylov solvers through the cusp::multiply() function.
 * Note that this operator() is templatized by the SolverVector type
 * to implement mixed-precision.
 */
template <typename PrecVector>
template <typename SolverVector>
void
Precond<PrecVector>::operator()(const SolverVector& v,
                                SolverVector& z)
{
    // If no preconditioner, copy RHS vector v into solution vector z and return.
    if (m_precondType == None) {
        cusp::blas::copy(v, z);
        return;
    }

    // Invoke the preconditioner solve function.
    cusp::blas::copy(v, m_vp);
    solve(m_vp, m_zp);
    cusp::blas::copy(m_zp, z);
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
        getSRev(z, m_buffer);
        rightTrans(m_buffer, z);
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

    if (m_ilu_level >= 0) {
        if (m_numPartitions > 1 && m_precondType == Spike) {
            if (m_variableBandwidth) {
                permute(rhs, m_secondReordering,m_buffer2);
                // Calculate modified RHS
                sparseSweep(rhs, rhs);

                permute(rhs, m_secondReordering, sol);

                // Solve reduced system
                partFullFwdSweep(sol);
                partFullBckSweep(sol);

                purifyRHS(sol, m_buffer2);
                permute(m_buffer2, m_secondPerm, sol);
            } else {
                sol = rhs;
                // Calculate modified RHS
                sparseSweep(rhs, rhs);

                // Solve reduced system
                partFullFwdSweep(rhs);
                partFullBckSweep(rhs);

                // Purify RHS
                purifyRHS(rhs, sol);
            }
        } else 
            sol = rhs;

        sparseSweep(sol, sol);
        return;
    }

    if (m_numPartitions > 1 && m_precondType == Spike) {
        if (m_variableBandwidth) {
            permute(rhs, m_secondReordering,m_buffer2);
            // Calculate modified RHS
            partBandedFwdSweep(rhs);
            partBandedBckSweep(rhs);

            permute(rhs, m_secondReordering, sol);

            // Solve reduced system
            partFullFwdSweep(sol);
            partFullBckSweep(sol);

            purifyRHS(sol, m_buffer2);
            permute(m_buffer2, m_secondPerm, sol);
        } else {
            sol = rhs;
            // Calculate modified RHS
            partBandedFwdSweep(rhs);
            partBandedBckSweep(rhs);

            // Solve reduced system
            partFullFwdSweep(rhs);
            partFullBckSweep(rhs);

            // Purify RHS
            purifyRHS(rhs, sol);
        }
    } else
        sol = rhs;

    // Get purified solution
    // if (m_numPartitions > CPU_GPU_SWEEP_THRESHOLD) {
        partBandedFwdSweep(sol);
        partBandedBckSweep(sol);
    // } else
        // partBandedSweepsH(sol);
}

/**
 * This function left transforms the system. We first apply the DB row
 * scaling and permutation (or only the DB row permutation) after which we
 * apply the RCM row permutation.
 */
template <typename PrecVector>
void
Precond<PrecVector>::leftTrans(PrecVector&  v,
                               PrecVector&  z)
{
    if (m_scale)
        scaleAndPermute(v, m_optPerm, m_dbRowScale, z);
    else
        permute(v, m_optPerm, z);
}

/**
 * This function right transforms the system. We apply the RCM column 
 * permutation and, if needed, the DB column scaling.
 */
template <typename PrecVector>
void
Precond<PrecVector>::rightTrans(PrecVector&  v,
                                PrecVector&  z)
{
    if (m_scale)
        permuteAndScale(v, m_optReordering, m_dbColScale, z);
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
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.begin(), thrust::make_permutation_iterator(scale.begin(), perm.begin()))), Multiply<PrecValueType>()),
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.end(), thrust::make_permutation_iterator(scale.end(), perm.end()))), Multiply<PrecValueType>()),
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
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.begin(), scale.begin())), Multiply<PrecValueType>()),
            thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(v.end(), scale.end())), Multiply<PrecValueType>()),
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
 *       row and column permutations obtained from the DB algorithm
 *   dbRowScale
 *   dbColScale
 *       row and column scaling factors obtained from the DB algorithm
 */
template <typename PrecVector>
template <typename Matrix>
void
Precond<PrecVector>::transformToBandedMatrix(const Matrix&  A)
{
    CPUTimer reorder_timer, assemble_timer, transfer_timer;

    transfer_timer.Start();

    // Reorder the matrix and apply drop-off. For this, we convert the
    // input matrix to CSR format and copy it on the host.
    PrecMatrixCsrH Acsrh;

#ifdef USE_OLD_CUSP
    Acsrh = A;
#else
    {
        PrecMatrixCooH Acooh = A;

        if (!Acooh.is_sorted_by_row()) {
            Acsrh.resize(A.num_rows, A.num_rows, A.num_entries);

            thrust::fill(Acsrh.row_offsets.begin(), Acsrh.row_offsets.end(), 0);

            for (int i = 0; i < Acooh.num_entries; i++)
                Acsrh.row_offsets[Acooh.row_indices[i]] ++;

            thrust::inclusive_scan(Acsrh.row_offsets.begin(), Acsrh.row_offsets.end(), Acsrh.row_offsets.begin());

            for (int i = Acooh.num_entries - 1; i >= 0; i--) {
                int idx = (--Acsrh.row_offsets[Acooh.row_indices[i]]);
                Acsrh.column_indices[idx] = Acooh.column_indices[i];
                Acsrh.values[idx]         = Acooh.values[i];
            }
        } else 
            Acsrh = Acooh;
    }
#endif

    {
        int num_entries = 0;
        IntVectorH   row_offsets(Acsrh.num_rows + 1, 0);
        for (int i = 0; i < Acsrh.num_rows; i++) {
            int start_idx = Acsrh.row_offsets[i], end_idx = Acsrh.row_offsets[i+1];
            for (int j = start_idx; j < end_idx; j++) {
                if (Acsrh.values[j] != 0) {
                    if (num_entries != j) {
                        Acsrh.column_indices[num_entries] = Acsrh.column_indices[j];
                        Acsrh.values[num_entries] = Acsrh.values[j];
                    }
                    num_entries ++;
                }
            }
            row_offsets[i + 1] = num_entries;
        }
        Acsrh.resize(Acsrh.num_rows, Acsrh.num_rows, num_entries);
        thrust::copy(row_offsets.begin(), row_offsets.end(), Acsrh.row_offsets.begin());
    }

    transfer_timer.Stop();
    m_time_transfer = transfer_timer.getElapsed();

    PrecVectorH  B;
    IntVectorH   optReordering;
    IntVectorH   optPerm;
    IntVectorH   secondReorder;
    IntVectorH   secondPerm;

    IntVector    dbRowPerm(m_n);

    Graph<PrecValueType>  graph(m_trackReordering);

    IntVectorH   offDiagPerms_left;
    IntVectorH   offDiagPerms_right;

    MatrixMapH   offDiagMap;
    MatrixMapH   WVMap;
    MatrixMapH   typeMap;
    MatrixMapH   bandedMatMap;
    MatrixMapFH  scaleMap;

    const int    BANDWIDTH_THRESHOLD = 64;
    bool         doRCM   = (m_ilu_level < 0 && (m_maxBandwidth > BANDWIDTH_THRESHOLD));
    bool         doSloan = (m_ilu_level >= 0);
    reorder_timer.Start();
    m_k_reorder = graph.reorder(Acsrh, m_testDB, m_doDB, m_dbFirstStageOnly, m_scale, doRCM, doSloan, optReordering, optPerm, dbRowPerm, m_dbRowScale, m_dbColScale, scaleMap, m_k_db);
    reorder_timer.Stop();

    m_time_DB        = graph.getTimeDB();
    m_time_DB_pre    = graph.getTimeDBPre();
    m_time_DB_first  = graph.getTimeDBFirst();
    m_time_DB_second = graph.getTimeDBSecond();
    m_time_DB_post   = graph.getTimeDBPost();
    m_time_reorder += reorder_timer.getElapsed();

    if (m_testDB)
        return;
    
    if (m_k_reorder > m_maxBandwidth || m_dropOff_frac > 0) {
        CPUTimer loc_timer;
        loc_timer.Start();
        m_k = graph.dropOff(m_dropOff_frac, m_maxBandwidth, m_dropOff_actual);
        loc_timer.Stop();

        m_time_dropOff = loc_timer.getElapsed();
    }
    else {
        m_dropOff_actual = 0;
        m_time_dropOff = 0;
        m_k = m_k_reorder;
    }

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
        PrecMatrixCooH Acooh;
        graph.assembleBandedMatrix(m_k, m_saveMem, m_numPartitions, m_ks_col_host, m_ks_row_host, Acooh,
                                   m_ks_host, m_BOffsets_host, 
                                   typeMap, bandedMatMap);
        assemble_timer.Stop();
        m_time_cpu_assemble += assemble_timer.getElapsed();

        m_timer.Start();

        if (m_ilu_level >= 0)
            graph.get_csr_matrix(m_Acsrh, m_numPartitions);
        else {
            if (m_gpuCount == 1) {
                PrecMatrixCoo Acoo = Acooh;
                m_B.resize(m_BOffsets_host[m_numPartitions], 0);
                int blockX = Acoo.num_entries, gridX = 1, gridY = 1;
                kernelConfigAdjust(blockX, gridX, gridY, BLOCK_SIZE, MAX_GRID_DIMENSION);
                dim3 grids(gridX, gridY);

                int*           d_rows = thrust::raw_pointer_cast(&(Acoo.row_indices[0]));
                int*           d_cols = thrust::raw_pointer_cast(&(Acoo.column_indices[0]));
                PrecValueType* d_vals = thrust::raw_pointer_cast(&(Acoo.values[0]));
                PrecValueType* dB     = thrust::raw_pointer_cast(&m_B[0]);

                m_ks = m_ks_host;
                m_BOffsets = m_BOffsets_host;

                int*           d_ks   = thrust::raw_pointer_cast(&m_ks[0]);
                int*       d_offsets  = thrust::raw_pointer_cast(&m_BOffsets[0]);

                device::var::copyFromCOOMatrixToBandedMatrix<<<grids, blockX>>>(Acoo.num_entries, d_ks, d_rows, d_cols, d_vals, dB, d_offsets, m_n / m_numPartitions, m_n % m_numPartitions, m_saveMem);
            } else {
                int cur_device = 0;
                cudaGetDevice(&cur_device);
                int rows_per_partition = m_n / m_numPartitions;
                int rows_remainder = m_n % m_numPartitions;
                int partitions_per_device = m_numPartitions / m_gpuCount;
                int part_remainder = m_numPartitions % m_gpuCount;
                int coo_idx_begin = 0;

                int partBegin = 0, partEnd;
                int rowBegin = 0;

                m_all_Bs.resize(m_gpuCount);
                m_all_BOffsets.resize(m_gpuCount);
                m_all_ks.resize(m_gpuCount);
                m_all_BOffsets_host.resize(m_gpuCount);
                m_all_ks_host.resize(m_gpuCount);
                m_all_num_rows.resize(m_gpuCount);
                m_all_num_partitions.resize(m_gpuCount);

                std::vector<PrecMatrixCooH> my_Acoohs(m_gpuCount);
                std::vector<PrecMatrixCoo>  my_Acoos(m_gpuCount);

                for (int i = 0; i < m_gpuCount; i++) {
                    cudaSetDevice(i);

                    if (i < part_remainder) {
                        partEnd = partBegin + (partitions_per_device + 1);
                    } else {
                        partEnd = partBegin + partitions_per_device;
                    }

                    int r_begin = rowBegin;
                    int rowEnd;
                    int b_offset_begin = m_BOffsets_host[partBegin];
                    int coo_idx_end;
                    size_t nnz = Acooh.row_indices.size();

                    m_all_ks_host[i].resize(partEnd - partBegin);
                    m_all_BOffsets_host[i].resize(partEnd - partBegin + 1);

                    for (int j = partBegin; j < partEnd; j++) {
                        if (j < rows_remainder) {
                            rowEnd = r_begin + (rows_per_partition + 1);
                        } else {
                            rowEnd = r_begin + rows_per_partition;
                        }
                        m_all_ks_host[i][j - partBegin] = m_ks_host[j];
                        m_all_BOffsets_host[i][j - partBegin] = m_BOffsets_host[j] - b_offset_begin;
                        r_begin = rowEnd;
                    }
                    m_all_BOffsets_host[i][partEnd - partBegin] = m_BOffsets_host[partEnd] - b_offset_begin;

                    m_all_Bs[i].resize(m_BOffsets_host[partEnd] - b_offset_begin);

                    m_all_ks[i] = m_all_ks_host[i];
                    m_all_BOffsets[i] = m_all_BOffsets_host[i];
                    m_all_num_rows[i] = rowEnd - rowBegin;
                    m_all_num_partitions[i] = partEnd - partBegin;

                    for (coo_idx_end = coo_idx_begin; coo_idx_end < nnz; coo_idx_end++) {
                        if (Acooh.row_indices[coo_idx_end] >= rowEnd) {
                            break;
                        }
                    }

                    my_Acoohs[i].resize(
                        rowEnd - rowBegin,
                        rowEnd - rowBegin,
                        coo_idx_end - coo_idx_begin
                    );

                    for (int j = coo_idx_begin; j < coo_idx_end; j++) {
                        my_Acoohs[i].row_indices[j - coo_idx_begin] = Acooh.row_indices[j] - rowBegin;
                        my_Acoohs[i].column_indices[j - coo_idx_begin] = Acooh.column_indices[j] - rowBegin;
                        my_Acoohs[i].values[j - coo_idx_begin] = Acooh.values[j];
                    }
                    my_Acoos[i] = my_Acoohs[i];

                    coo_idx_begin = coo_idx_end;
                    rowBegin  = rowEnd;
                    partBegin = partEnd;
                }

                omp_set_num_threads(m_gpuCount);

                int gpu_count = m_gpuCount;

#pragma omp parallel for shared(gpu_count)
                for (int i = 0; i < gpu_count; i++) {
                    cudaSetDevice(i);
                    int blockX = my_Acoos[i].num_entries, gridX = 1, gridY = 1;
                    kernelConfigAdjust(blockX, gridX, gridY, BLOCK_SIZE, MAX_GRID_DIMENSION);
                    dim3 grids(gridX, gridY);

                    int*           d_rows = thrust::raw_pointer_cast(&(my_Acoos[i].row_indices[0]));
                    int*           d_cols = thrust::raw_pointer_cast(&(my_Acoos[i].column_indices[0]));
                    PrecValueType* d_vals = thrust::raw_pointer_cast(&(my_Acoos[i].values[0]));
                    PrecValueType* dB     = thrust::raw_pointer_cast(&m_all_Bs[i][0]);

                    int*           d_ks   = thrust::raw_pointer_cast(&m_all_ks[i][0]);
                    int*       d_offsets  = thrust::raw_pointer_cast(&m_all_BOffsets[i][0]);

                    device::var::copyFromCOOMatrixToBandedMatrix<<<grids, blockX>>>(my_Acoos[i].num_entries, d_ks, d_rows, d_cols, d_vals, dB, d_offsets, m_all_num_rows[i] / m_all_num_partitions[i], m_all_num_rows[i] % m_all_num_partitions[i], m_saveMem);
                }

                cudaSetDevice(cur_device);
            }
        }

        m_timer.Stop();
        m_time_toBanded = m_timer.getElapsed();
    } else {
        PrecMatrixCooH Acooh;
        assemble_timer.Start();

        if (m_numPartitions > 1) {
            graph.assembleOffDiagMatrices(m_k, m_numPartitions, m_WV_host, m_offDiags_host, m_offDiagWidths_left_host, m_offDiagWidths_right_host, offDiagPerms_left, offDiagPerms_right, typeMap, offDiagMap, WVMap);
            graph.assembleBandedMatrix(m_k, m_saveMem, m_numPartitions, m_ks_col_host, m_ks_row_host, Acooh,
                    m_ks_host, m_BOffsets_host, 
                    typeMap, bandedMatMap);
        } else {
            graph.assembleBandedMatrix(m_k, m_ks_col_host, m_ks_row_host, Acooh, typeMap, bandedMatMap);
            m_ks_host.resize(1, m_k);
            m_BOffsets_host.resize(1, 0);
        }
        assemble_timer.Stop();
        m_time_cpu_assemble += assemble_timer.getElapsed();


        if (m_ilu_level < 0) {
            transfer_timer.Start();
            PrecMatrixCoo Acoo = Acooh;
            if (m_saveMem)
                m_B.resize((size_t)(m_k + 1) * m_n);
            else
                m_B.resize((size_t)(2 * m_k + 1) * m_n);

            transfer_timer.Stop();
            m_time_transfer += transfer_timer.getElapsed();

            m_timer.Start();
            int blockX = Acoo.num_entries, gridX = 1, gridY = 1;
            kernelConfigAdjust(blockX, gridX, gridY, BLOCK_SIZE, MAX_GRID_DIMENSION);
            dim3 grids(gridX, gridY);

            int*           d_rows = thrust::raw_pointer_cast(&(Acoo.row_indices[0]));
            int*           d_cols = thrust::raw_pointer_cast(&(Acoo.column_indices[0]));
            PrecValueType* d_vals = thrust::raw_pointer_cast(&(Acoo.values[0]));
            PrecValueType* dB     = thrust::raw_pointer_cast(&m_B[0]);

            device::copyFromCOOMatrixToBandedMatrix<<<grids, blockX>>>(Acoo.num_entries, m_k, d_rows, d_cols, d_vals, dB, m_saveMem);
            m_timer.Stop();
            m_time_toBanded = m_timer.getElapsed();
        } else  {
            m_timer.Start();
            graph.get_csr_matrix(m_Acsrh, m_numPartitions);
            m_timer.Stop();
            m_time_toBanded = m_timer.getElapsed();
        }
    }

    transfer_timer.Start();

    // Copy the banded matrix and permutation data to the device.
    m_optReordering = optReordering;
    m_optPerm = optPerm;

    if (m_variableBandwidth || (m_numPartitions > 1 && m_ilu_level >= 0)) {
        m_offDiagWidths_left = m_offDiagWidths_left_host;
        m_offDiagWidths_right = m_offDiagWidths_right_host;
        m_offDiagPerms_left = offDiagPerms_left;
        m_offDiagPerms_right = offDiagPerms_right;

        m_spike_ks.resize(m_numPartitions - 1);
        m_ROffsets.resize(m_numPartitions - 1);
        cusp::blas::fill(m_spike_ks, m_k);
        thrust::sequence(m_ROffsets.begin(), m_ROffsets.end(), 0, 4*m_k*m_k);
    }

    if (m_variableBandwidth) {
        m_WVOffsets.resize(m_numPartitions - 1);
        m_compB2Offsets.resize(m_numPartitions - 1);
        m_partialBOffsets.resize(m_numPartitions-1);

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
        IntVector buffer2(m_n);
        combinePermutation(dbRowPerm, m_optPerm, buffer2);
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
    device::copyFromCOOMatrixToBandedMatrix<<<grids, blockX>>>(nnz, m_k, d_rows, d_cols, d_vals, dB, m_saveMem);
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
    if (m_variableBandwidth || m_ilu_level >= 0) {
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
 *         Precond::partFullLU_var()
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
#if 0
    if (m_variableBandwidth || m_ilu_level >= 0)
        partBlockedFullLU_var();
    else
        partFullLU_const();
#endif
    if (!m_variableBandwidth && m_ilu_level < 0) {
        m_spike_ks.resize(m_numPartitions - 1);
        m_ROffsets.resize(m_numPartitions - 1);
        cusp::blas::fill(m_spike_ks, m_k);
        thrust::sequence(m_ROffsets.begin(), m_ROffsets.end(), 0, 4*m_k*m_k);
    }
    partBlockedFullLU_var();
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

            if (i == m_k) {
                if(threads > 1024)
                    device::var::fullLU_div_safe_general<PrecValueType><<<m_numPartitions-1, 512>>>(d_R, p_spike_ks,  p_ROffsets, i);
                else
                    device::var::fullLU_div_safe<PrecValueType><<<m_numPartitions-1, threads>>>(d_R, p_spike_ks,  p_ROffsets, i);
            }

            if (threads > 1024)
                device::var::fullLU_sub_div_safe_general<PrecValueType><<<grids, 512>>>(d_R, p_spike_ks,  p_ROffsets, i);
            else
                device::var::fullLU_sub_div_safe<PrecValueType><<<grids, threads>>>(d_R, p_spike_ks,  p_ROffsets, i);
        }
    } else {
        for(int i = m_k; i < two_k-1; i++) {
            int  threads = two_k-1-i;
            dim3 grids(two_k-1-i, m_numPartitions-1);

            if (i == m_k) {
                if(threads > 1024)
                    device::var::fullLU_div_general<PrecValueType><<<m_numPartitions-1, 512>>>(d_R, p_spike_ks,  p_ROffsets, i);
                else
                    device::var::fullLU_div<PrecValueType><<<m_numPartitions-1, threads>>>(d_R, p_spike_ks,  p_ROffsets, i);
            }

            if (threads > 1024)
                device::var::fullLU_sub_div_general<PrecValueType><<<grids, 512>>>(d_R, p_spike_ks,  p_ROffsets, i);
            else
                device::var::fullLU_sub_div<PrecValueType><<<grids, threads>>>(d_R, p_spike_ks,  p_ROffsets, i);
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

template <typename PrecVector>
void
Precond<PrecVector>::partBlockedFullLU_var()
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
    const int BLOCK_FACTOR = 8;
    for(int i = m_k; i < two_k-1; i += BLOCK_FACTOR) {
        int  left_rows = two_k - i;
        int  threads = (two_k-1-i) * (left_rows < BLOCK_FACTOR ? (left_rows - 1) : (BLOCK_FACTOR - 1));

        dim3 grids(two_k-BLOCK_FACTOR-i, m_numPartitions-1);

        if (m_safeFactorization) {
            if(threads > 1024)
                device::var::blockedFullLU_phase1_safe_general<PrecValueType><<<m_numPartitions-1, 512>>>(d_R, p_spike_ks,  p_ROffsets, i, left_rows < BLOCK_FACTOR ? left_rows : BLOCK_FACTOR);
            else
                device::var::blockedFullLU_phase1_safe_general<PrecValueType><<<m_numPartitions-1, threads>>>(d_R, p_spike_ks,  p_ROffsets, i, left_rows < BLOCK_FACTOR ? left_rows : BLOCK_FACTOR);
        } else {
            if(threads > 1024)
                device::var::blockedFullLU_phase1_general<PrecValueType><<<m_numPartitions-1, 512>>>(d_R, p_spike_ks,  p_ROffsets, i, left_rows < BLOCK_FACTOR ? left_rows : BLOCK_FACTOR);
            else
                device::var::blockedFullLU_phase1_general<PrecValueType><<<m_numPartitions-1, threads>>>(d_R, p_spike_ks,  p_ROffsets, i, left_rows < BLOCK_FACTOR ? left_rows : BLOCK_FACTOR);
        }

        if (left_rows <= BLOCK_FACTOR)
            break;

        device::var::blockedFullLU_phase2_general<PrecValueType><<<grids, BLOCK_FACTOR, BLOCK_FACTOR * sizeof(PrecValueType)>>>(d_R, p_spike_ks, p_ROffsets, i, BLOCK_FACTOR);

        threads = two_k - BLOCK_FACTOR - i;

        if (threads > 1024)
            device::var::blockedFullLU_phase3_general<PrecValueType><<<grids, 512, sizeof(PrecValueType) * BLOCK_FACTOR>>>(d_R, p_spike_ks, p_ROffsets, i, BLOCK_FACTOR);
        else
            device::var::blockedFullLU_phase3_general<PrecValueType><<<grids, threads, sizeof(PrecValueType) * BLOCK_FACTOR>>>(d_R, p_spike_ks, p_ROffsets, i, BLOCK_FACTOR);
    }

    {
        dim3 grids(m_k-1, m_numPartitions-1);
        if (m_k >= 1024)
            device::var::fullLU_post_divide_general<PrecValueType><<<grids, 512>>>(d_R, p_spike_ks, p_ROffsets);
        else
            device::var::fullLU_post_divide<PrecValueType><<<grids, m_k-1>>>(d_R, p_spike_ks, p_ROffsets);
    }
}

/*! \brief This function does incomplete LU to the provided
 * CSR matrix.
 *
 * The integer p specifies the filling-in factor and tau
 * indicates the threshold of drop-off.
 */
template <typename PrecVector>
void
Precond<PrecVector>::ILUT(PrecMatrixCsrH &Acsrh, int p, PrecValueType tau)
{
    int numPartitions = m_numPartitions;
    int num_threads = std::min(16, numPartitions);

    omp_set_num_threads(num_threads);

    IntVectorH    lu_row_offsets(m_n + 1, 0);
    std::vector<IntVectorH>    lu_column_indices(numPartitions);
    std::vector<PrecVectorH>   lu_values(numPartitions);
    PrecVectorH   wvector(m_n, 0);
    IntVectorH    in_wvector(m_n, 0);

    IntVectorH pivot_positions(m_n, -1);
    IntVectorH l_columns(m_n), u_columns(m_n);
    PrecVectorH l_values(m_n), u_values(m_n);
    IntVectorH w_nonzeros(m_n);

    int remainder = m_n % numPartitions;
    int partSize  = m_n / numPartitions;

//#pragma omp parallel for shared (numPartitions, remainder, partSize, lu_row_offsets, wvector, in_wvector, pivot_positions, l_columns, u_columns, l_values, u_values, w_nonzeros)
    for (int q = 0; q < numPartitions; q++) {
        int start_row = 0, end_row = 0;

        PrecVectorH& loc_lu_values         = lu_values[q];
        IntVectorH&  loc_lu_column_indices = lu_column_indices[q];

        if (q < remainder) {
            start_row = q * (partSize + 1);
            end_row   = start_row + (partSize + 1);
        }
        else {
            start_row = q * partSize + remainder;
            end_row   = start_row + partSize;
        }

        {
            int start_idx = Acsrh.row_offsets[start_row];
            int end_idx = Acsrh.row_offsets[start_row + 1];

            lu_row_offsets[start_row]     = 0;
            lu_row_offsets[start_row + 1] = (end_idx - start_idx);

            loc_lu_column_indices.insert(loc_lu_column_indices.end(), Acsrh.column_indices.begin() + start_idx, Acsrh.column_indices.begin() + end_idx);
            loc_lu_values.insert(loc_lu_values.end(), Acsrh.values.begin() + start_idx, Acsrh.values.begin() + end_idx);

            int l;
            for (l = start_idx; l < end_idx; l++)
                if (Acsrh.column_indices[l] == start_row) {
                    pivot_positions[start_row] = l - start_idx;
                    m_pivots[start_row] = Acsrh.values[l];
                    break;
                }
        }

        int wvec_size;

        int l_size, u_size;

        for (int i = start_row + 1; i < end_row; i++) {
            int start_idx = Acsrh.row_offsets[i];
            int end_idx = Acsrh.row_offsets[i+1];

            int nl = 0, nu = 0;

            PrecValueType tau_i = (PrecValueType)0;

            wvec_size = start_row + end_idx - start_idx;

            std::priority_queue<int, std::vector<int>, std::greater<int> > pq;
            for (int l = start_idx; l < end_idx; l++) {
                PrecValueType tmp_val = Acsrh.values[l];
                int cur_k = Acsrh.column_indices[l];
                wvector[cur_k] = tmp_val;
                in_wvector[cur_k] = l - start_idx + 1;
                tau_i += tmp_val * tmp_val;
                w_nonzeros[start_row + l - start_idx] = cur_k;
                if (cur_k < i) {
                    pq.push(cur_k);
                    nl ++;
                } else  if (cur_k > i)
                    nu ++;
            }
            tau_i = sqrt(tau_i) / (end_idx - start_idx) * tau;

            while (!pq.empty()) {
                int cur_k = pq.top();
                pq.pop();

                int start_k_idx = lu_row_offsets[cur_k];
                int end_k_idx = lu_row_offsets[cur_k+1];

                int l2 = pivot_positions[cur_k];
                PrecValueType val_i_k = (PrecValueType)0;

                if (fabs(loc_lu_values[l2]) < BURST_VALUE) {
                    if (loc_lu_values[l2] < 0)
                        loc_lu_values[l2] = -BURST_VALUE;
                    else
                        loc_lu_values[l2] = BURST_VALUE;
                }

                val_i_k = (wvector[cur_k] /= loc_lu_values[l2]);

                // Applying drop-off to w[cur_k]
                if (fabs(val_i_k) < tau_i) {
                    in_wvector[cur_k] = 0;
                    continue;
                }

                for (l2++; l2 < end_k_idx; l2++) {
                    int tar_j = loc_lu_column_indices[l2];

                    wvector[tar_j] -= val_i_k * loc_lu_values[l2];

                    if(!in_wvector[tar_j]) {
                        w_nonzeros[wvec_size++] = tar_j;
                        in_wvector[tar_j] = wvec_size;
                        if (tar_j < i)
                            pq.push(tar_j);
                    }
                }
            } // end while

            // Apply drop-off to wvector
            {
                l_size = u_size = start_row;
                for (int w_it = start_row; w_it < wvec_size; w_it++) {
                    int cur_k = w_nonzeros[w_it];
                    if (!in_wvector[cur_k])
                        continue;
                    PrecValueType tmp_val = wvector[cur_k];

                    if (cur_k == i) {
                        m_pivots[i] = ((tmp_val > BURST_VALUE || tmp_val < -BURST_VALUE) ? tmp_val : (tmp_val > 0 ? BURST_VALUE : -BURST_VALUE));
                        continue;
                    }

                    if (fabs(tmp_val) < tau_i)
                        continue;

                    if (cur_k < i) {
                        l_columns[l_size] = cur_k;
                        l_values[l_size] = tmp_val;
                        l_size++;
                    } else {
                        u_columns[u_size] = cur_k;
                        u_values[u_size] = tmp_val;
                        u_size++;
                    }
                }

                // Clear the content of wvector and in_wvector for usage of next iteration
                for (int w_it = start_row; w_it < wvec_size; w_it++) {
                    int cur_k = w_nonzeros[w_it];
                    wvector[cur_k] = (PrecValueType)0;
                    in_wvector[cur_k] = 0;
                }

                if (l_size - start_row > p + nl) {
                    findPthMax(l_columns.begin() + start_row, l_columns.begin() + l_size, 
                            l_values.begin() + start_row,  l_values.begin() + l_size,
                            p + nl);
                    l_size = p + nl + start_row;
                }

                if (u_size - start_row > p + nu) {
                    findPthMax(u_columns.begin() + start_row, u_columns.begin() + u_size, 
                            u_values.begin() + start_row,  u_values.begin() + u_size,
                            p + nu);
                    u_size = p + nu + start_row;
                }

                loc_lu_column_indices.insert(loc_lu_column_indices.end(), l_columns.begin() + start_row, l_columns.begin() + l_size);
                loc_lu_values.insert(loc_lu_values.end(), l_values.begin() + start_row, l_values.begin() + l_size);

                loc_lu_column_indices.push_back(i);
                loc_lu_values.push_back(m_pivots[i]);
                pivot_positions[i] = loc_lu_column_indices.size() - 1;

                loc_lu_column_indices.insert(loc_lu_column_indices.end(), u_columns.begin() + start_row, u_columns.begin() + u_size);
                loc_lu_values.insert(loc_lu_values.end(), u_values.begin() + start_row, u_values.begin() + u_size);

                if (i != end_row - 1)
                    lu_row_offsets[i+1] = loc_lu_column_indices.size();
            }

        } // end for
    } // end for

    int new_nnz = lu_column_indices[0].size();
    for (int q = 1; q < numPartitions; q++) {
        int start_row = 0, end_row = 0;

        if (q < remainder) {
            start_row = q * (partSize + 1);
            end_row   = start_row + (partSize + 1);
        }
        else {
            start_row = q * partSize + remainder;
            end_row   = start_row + partSize;
        }

        for (int i = start_row; i < end_row; i++)
            lu_row_offsets[i] += new_nnz;

        new_nnz += lu_column_indices[q].size();

        lu_column_indices[0].insert(lu_column_indices[0].end(), lu_column_indices[q].begin(), lu_column_indices[q].end());
        lu_values[0].insert(lu_values[0].end(), lu_values[q].begin(), lu_values[q].end());
    }
    lu_row_offsets[m_n] = new_nnz;

    IntVectorH& final_lu_column_indices = lu_column_indices[0];
    PrecVectorH& final_lu_values        = lu_values[0];

    Acsrh.resize(m_n, m_n, new_nnz);
    cusp::blas::fill(Acsrh.row_offsets, 0);

    IntVectorH  tmp_row_indices(new_nnz);
    IntVectorH  tmp_column_indices(new_nnz);
    PrecVectorH tmp_values(new_nnz);

    for (int i = 0; i < new_nnz; i++)
        Acsrh.row_offsets[final_lu_column_indices[i]] ++;

    thrust::exclusive_scan(Acsrh.row_offsets.begin(), Acsrh.row_offsets.end(), Acsrh.row_offsets.begin());

    for (int i = 0; i < m_n; i++) {
        int start_idx = lu_row_offsets[i];
        int end_idx   = lu_row_offsets[i+1];

        for (int l = start_idx; l < end_idx; l++) {
            int cur_k = final_lu_column_indices[l];
            int idx = Acsrh.row_offsets[cur_k];
            Acsrh.row_offsets[cur_k] ++;
            tmp_row_indices[idx] = i;
            tmp_column_indices[idx] = cur_k; 
            tmp_values[idx] = final_lu_values[l]; 
        }
    }

    cusp::blas::fill(Acsrh.row_offsets, 0);
    for (int i = 0; i < new_nnz; i++)
        Acsrh.row_offsets[tmp_row_indices[i]] ++;

    thrust::inclusive_scan(Acsrh.row_offsets.begin(), Acsrh.row_offsets.end(), Acsrh.row_offsets.begin());

    for (int i = new_nnz - 1; i >= 0; i--) {
        int idx = --Acsrh.row_offsets[tmp_row_indices[i]];
        Acsrh.column_indices[idx] = tmp_column_indices[i];
        Acsrh.values[idx] = tmp_values[i];
    }

    cusp::blas::copy(lu_row_offsets, Acsrh.row_offsets);
}

/*! \brief This function does incomplete LU to all but the
 * last one partition and incomplete UL to all but the first
 * one partition of the matrix Acsrh.
 *
 * The integer p specifies the filling-in factor and tau
 * indicates the threshold of drop-off.
 */
template <typename PrecVector>
void
Precond<PrecVector>::ILUULT(PrecMatrixCsrH &Acsrh, PrecMatrixCsrH &Acsrh2, int p, PrecValueType tau)
{
    int numPartitions = m_numPartitions;

    if (numPartitions == 2 || numPartitions == 4)
        omp_set_num_threads(2);
    else
        omp_set_num_threads(1);

    IntVectorH    lu_row_offsets(m_n + 1, 0);
    std::vector<IntVectorH>    lu_column_indices(numPartitions - 1);
    std::vector<PrecVectorH>   lu_values(numPartitions - 1);
    IntVectorH    ul_row_offsets(m_n + 1, 0);
    std::vector<IntVectorH>    ul_column_indices(numPartitions - 1);
    std::vector<PrecVectorH>   ul_values(numPartitions - 1);
    PrecVectorH   wvector_lu(m_n, 0);
    IntVectorH    in_wvector_lu(m_n, 0);
    PrecVectorH   wvector_ul(m_n, 0);
    IntVectorH    in_wvector_ul(m_n, 0);

    IntVectorH pivot_positions_lu(m_n, -1);
    IntVectorH pivot_positions_ul(m_n, -1);
    IntVectorH l_columns_lu(m_n), u_columns_lu(m_n);
    PrecVectorH l_values_lu(m_n), u_values_lu(m_n);
    IntVectorH w_nonzeros_lu(m_n);
    IntVectorH l_columns_ul(m_n), u_columns_ul(m_n);
    PrecVectorH l_values_ul(m_n), u_values_ul(m_n);
    IntVectorH w_nonzeros_ul(m_n);

    int remainder = m_n % numPartitions;
    int partSize  = m_n / numPartitions;

    int it_max = 2 * numPartitions - 2;

#pragma omp parallel for shared (it_max, numPartitions, remainder, partSize, lu_row_offsets, ul_row_offsets, wvector_lu, wvector_ul, in_wvector_lu, in_wvector_ul, pivot_positions_lu, pivot_positions_ul, l_columns_lu, l_columns_ul, u_columns_lu, u_columns_ul, l_values_lu, l_values_ul, u_values_lu, u_values_ul, w_nonzeros_lu, w_nonzeros_ul)
    for (int ori_q = 0; ori_q < it_max; ori_q++) {
        int start_row = 0, end_row = 0;

        if (ori_q < numPartitions - 1) {

            int q = ori_q;

            PrecVectorH& loc_lu_values         = lu_values[q];
            IntVectorH&  loc_lu_column_indices = lu_column_indices[q];

            if (q < remainder) {
                start_row = q * (partSize + 1);
                end_row   = start_row + (partSize + 1);
            }
            else {
                start_row = q * partSize + remainder;
                end_row   = start_row + partSize;
            }

            {
                int start_idx = Acsrh.row_offsets[start_row];
                int end_idx = Acsrh.row_offsets[start_row + 1];

                lu_row_offsets[start_row]     = 0;
                lu_row_offsets[start_row + 1] = (end_idx - start_idx);

                loc_lu_column_indices.insert(loc_lu_column_indices.end(), Acsrh.column_indices.begin() + start_idx, Acsrh.column_indices.begin() + end_idx);
                loc_lu_values.insert(loc_lu_values.end(), Acsrh.values.begin() + start_idx, Acsrh.values.begin() + end_idx);

                int l;
                for (l = start_idx; l < end_idx; l++)
                    if (Acsrh.column_indices[l] == start_row) {
                        pivot_positions_lu[start_row] = l - start_idx;
                        m_pivots[start_row] = Acsrh.values[l];
                        break;
                }
            }

            int wvec_size;

            int l_size, u_size;

            for (int i = start_row + 1; i < end_row; i++) {
                int start_idx = Acsrh.row_offsets[i];
                int end_idx = Acsrh.row_offsets[i+1];

                int nl = 0, nu = 0;

                PrecValueType tau_i = (PrecValueType)0;

                wvec_size = start_row + end_idx - start_idx;

                std::priority_queue<int, std::vector<int>, std::greater<int> > pq;
                for (int l = start_idx; l < end_idx; l++) {
                    PrecValueType tmp_val = Acsrh.values[l];
                    int cur_k = Acsrh.column_indices[l];
                    wvector_lu[cur_k] = tmp_val;
                    in_wvector_lu[cur_k] = l - start_idx + 1;
                    tau_i += tmp_val * tmp_val;
                    w_nonzeros_lu[start_row + l - start_idx] = cur_k;
                    if (cur_k < i) {
                        pq.push(cur_k);
                        nl ++;
                    } else  if (cur_k > i)
                        nu ++;
                }
                tau_i = sqrt(tau_i) / (end_idx - start_idx) * tau;

                while (!pq.empty()) {
                    int cur_k = pq.top();
                    pq.pop();

                    int start_k_idx = lu_row_offsets[cur_k];
                    int end_k_idx = lu_row_offsets[cur_k+1];

                    int l2 = pivot_positions_lu[cur_k];
                    PrecValueType val_i_k = (PrecValueType)0;

                    if (fabs(loc_lu_values[l2]) < BURST_VALUE) {
                        if (loc_lu_values[l2] < 0)
                            loc_lu_values[l2] = -BURST_VALUE;
                        else
                            loc_lu_values[l2] = BURST_VALUE;
                    }

                    val_i_k = (wvector_lu[cur_k] /= loc_lu_values[l2]);

                    // Applying drop-off to w[cur_k]
                    if (fabs(val_i_k) < tau_i) {
                        in_wvector_lu[cur_k] = 0;
                        continue;
                    }

                    for (l2++; l2 < end_k_idx; l2++) {
                        int tar_j = loc_lu_column_indices[l2];

                        wvector_lu[tar_j] -= val_i_k * loc_lu_values[l2];

                        if(!in_wvector_lu[tar_j]) {
                            w_nonzeros_lu[wvec_size++] = tar_j;
                            in_wvector_lu[tar_j] = wvec_size;
                            if (tar_j < i)
                                pq.push(tar_j);
                        }
                    }
                } // end while

                // Apply drop-off to wvector
                {
                    l_size = u_size = start_row;
                    for (int w_it = start_row; w_it < wvec_size; w_it++) {
                        int cur_k = w_nonzeros_lu[w_it];
                        if (!in_wvector_lu[cur_k])
                            continue;
                        PrecValueType tmp_val = wvector_lu[cur_k];

                        if (cur_k == i) {
                            m_pivots[i] = ((tmp_val > BURST_VALUE || tmp_val < -BURST_VALUE) ? tmp_val : (tmp_val > 0 ? BURST_VALUE : -BURST_VALUE));
                            continue;
                        }

                        if (fabs(tmp_val) < tau_i)
                            continue;

                        if (cur_k < i) {
                            l_columns_lu[l_size] = cur_k;
                            l_values_lu[l_size] = tmp_val;
                            l_size++;
                        } else {
                            u_columns_lu[u_size] = cur_k;
                            u_values_lu[u_size] = tmp_val;
                            u_size++;
                        }
                    }

                    // Clear the content of wvector and in_wvector for usage of next iteration
                    for (int w_it = start_row; w_it < wvec_size; w_it++) {
                        int cur_k = w_nonzeros_lu[w_it];
                        wvector_lu[cur_k] = (PrecValueType)0;
                        in_wvector_lu[cur_k] = 0;
                    }

                    if (l_size - start_row > p + nl) {
                        findPthMax(l_columns_lu.begin() + start_row, l_columns_lu.begin() + l_size, 
                                l_values_lu.begin() + start_row,  l_values_lu.begin() + l_size,
                                p + nl);
                        l_size = p + nl + start_row;
                    }

                    if (u_size - start_row > p + nu) {
                        findPthMax(u_columns_lu.begin() + start_row, u_columns_lu.begin() + u_size, 
                                u_values_lu.begin() + start_row,  u_values_lu.begin() + u_size,
                                p + nu);
                        u_size = p + nu + start_row;
                    }

                    loc_lu_column_indices.insert(loc_lu_column_indices.end(), l_columns_lu.begin() + start_row, l_columns_lu.begin() + l_size);
                    loc_lu_values.insert(loc_lu_values.end(), l_values_lu.begin() + start_row, l_values_lu.begin() + l_size);

                    loc_lu_column_indices.push_back(i);
                    loc_lu_values.push_back(m_pivots[i]);
                    pivot_positions_lu[i] = loc_lu_column_indices.size() - 1;

                    loc_lu_column_indices.insert(loc_lu_column_indices.end(), u_columns_lu.begin() + start_row, u_columns_lu.begin() + u_size);
                    loc_lu_values.insert(loc_lu_values.end(), u_values_lu.begin() + start_row, u_values_lu.begin() + u_size);

                    if (i != end_row - 1)
                        lu_row_offsets[i+1] = loc_lu_column_indices.size();
                }

            } // end for
        }  else {

            int q = ori_q - numPartitions + 2;

            PrecVectorH& loc_ul_values         = ul_values[q - 1];
            IntVectorH&  loc_ul_column_indices = ul_column_indices[q - 1];

            if (q < remainder) {
                start_row = q * (partSize + 1);
                end_row   = start_row + (partSize + 1);
            }
            else {
                start_row = q * partSize + remainder;
                end_row   = start_row + partSize;
            }

            {
                int start_idx = Acsrh.row_offsets[end_row - 1];
                int end_idx = Acsrh.row_offsets[end_row];

                ul_row_offsets[end_row - 1] = 0;
                ul_row_offsets[end_row - 2] = (end_idx - start_idx);

                loc_ul_column_indices.insert(loc_ul_column_indices.end(), Acsrh.column_indices.begin() + start_idx, Acsrh.column_indices.begin() + end_idx);
                loc_ul_values.insert(loc_ul_values.end(), Acsrh.values.begin() + start_idx, Acsrh.values.begin() + end_idx);

                int l;
                for (l = start_idx; l < end_idx; l++)
                    if (Acsrh.column_indices[l] == end_row - 1) {
                        pivot_positions_ul[end_row - 1] = l - start_idx;
                        m_pivots_ul[end_row - 1] = Acsrh.values[l];
                        break;
                    }
            }


            int wvec_size;

            int l_size, u_size;

            for (int i = end_row - 2; i >= start_row; i--) {
                int start_idx = Acsrh.row_offsets[i];
                int end_idx = Acsrh.row_offsets[i+1];

                int nl = 0, nu = 0;

                PrecValueType tau_i = (PrecValueType)0;

                wvec_size = start_row + end_idx - start_idx;

                std::priority_queue<int, std::vector<int> > pq;
                for (int l = start_idx; l < end_idx; l++) {
                    PrecValueType tmp_val = Acsrh.values[l];
                    int cur_k = Acsrh.column_indices[l];
                    wvector_ul[cur_k] = tmp_val;
                    in_wvector_ul[cur_k] = l - start_idx + 1;
                    tau_i += tmp_val * tmp_val;
                    w_nonzeros_ul[start_row + l - start_idx] = cur_k;
                    if (cur_k < i) 
                        nl ++;
                    else  if (cur_k > i) {
                        pq.push(cur_k);
                        nu ++;
                    }
                }
                tau_i = sqrt(tau_i) / (end_idx - start_idx) * tau;

                while (!pq.empty()) {
                    int cur_k = pq.top();
                    pq.pop();

                    int start_k_idx = ul_row_offsets[cur_k];
                    int end_k_idx = ul_row_offsets[cur_k-1];

                    int l2 = pivot_positions_ul[cur_k];
                    PrecValueType val_i_k = (PrecValueType)0;

                    if (fabs(loc_ul_values[l2]) < BURST_VALUE) {
                        if (loc_ul_values[l2] < 0)
                            loc_ul_values[l2] = -BURST_VALUE;
                        else
                            loc_ul_values[l2] = BURST_VALUE;
                    }

                    val_i_k = (wvector_ul[cur_k] /= loc_ul_values[l2]);

                    // Applying drop-off to w[cur_k]
                    if (fabs(val_i_k) < tau_i) {
                        in_wvector_ul[cur_k] = 0;
                        continue;
                    }

                    for (l2--; l2 >= start_k_idx; l2--) {
                        int tar_j = loc_ul_column_indices[l2];

                        wvector_ul[tar_j] -= val_i_k * loc_ul_values[l2];

                        if(!in_wvector_ul[tar_j]) {
                            w_nonzeros_ul[wvec_size++] = tar_j;
                            in_wvector_ul[tar_j] = wvec_size;
                            if (tar_j > i)
                                pq.push(tar_j);
                        }
                    }
                } // end while

                // Apply drop-off to wvector
                {
                    l_size = u_size = start_row;
                    for (int w_it = start_row; w_it < wvec_size; w_it++) {
                        int cur_k = w_nonzeros_ul[w_it];
                        if (!in_wvector_ul[cur_k])
                            continue;
                        PrecValueType tmp_val = wvector_ul[cur_k];

                        if (cur_k == i) {
                            m_pivots_ul[i] = ((tmp_val > BURST_VALUE || tmp_val < -BURST_VALUE) ? tmp_val : (tmp_val > 0 ? BURST_VALUE : -BURST_VALUE));
                            continue;
                        }

                        if (fabs(tmp_val) < tau_i)
                            continue;

                        if (cur_k < i) {
                            l_columns_ul[l_size] = cur_k;
                            l_values_ul[l_size] = tmp_val;
                            l_size++;
                        } else {
                            u_columns_ul[u_size] = cur_k;
                            u_values_ul[u_size] = tmp_val;
                            u_size++;
                        }
                    }

                    // Clear the content of wvector and in_wvector for usage of next iteration
                    for (int w_it = start_row; w_it < wvec_size; w_it++) {
                        int cur_k = w_nonzeros_ul[w_it];
                        wvector_ul[cur_k] = (PrecValueType)0;
                        in_wvector_ul[cur_k] = 0;
                    }

                    if (l_size - start_row > p + nl) {
                        findPthMax(l_columns_ul.begin() + start_row, l_columns_ul.begin() + l_size, 
                                l_values_ul.begin() + start_row,  l_values_ul.begin() + l_size,
                                p + nl);
                        l_size = p + nl + start_row;
                    }

                    if (u_size - start_row > p + nu) {
                        findPthMax(u_columns_ul.begin() + start_row, u_columns_ul.begin() + u_size, 
                                u_values_ul.begin() + start_row,  u_values_ul.begin() + u_size,
                                p + nu);
                        u_size = p + nu + start_row;
                    }

                    loc_ul_column_indices.insert(loc_ul_column_indices.end(), l_columns_ul.begin() + start_row, l_columns_ul.begin() + l_size);
                    loc_ul_values.insert(loc_ul_values.end(), l_values_ul.begin() + start_row, l_values_ul.begin() + l_size);

                    loc_ul_column_indices.push_back(i);
                    loc_ul_values.push_back(m_pivots_ul[i]);
                    pivot_positions_ul[i] = loc_ul_column_indices.size() - 1;

                    loc_ul_column_indices.insert(loc_ul_column_indices.end(), u_columns_ul.begin() + start_row, u_columns_ul.begin() + u_size);
                    loc_ul_values.insert(loc_ul_values.end(), u_values_ul.begin() + start_row, u_values_ul.begin() + u_size);

                    if (i != start_row)
                        ul_row_offsets[i-1] = loc_ul_column_indices.size();
                }
            } // end for
        } // end else
    } // end for

    int new_nnz    = lu_column_indices[0].size();
    int new_nnz_ul = ul_column_indices[numPartitions - 2].size();
    for (int q = 1; q < numPartitions - 1; q++) {
        int start_row = 0, end_row = 0;

        if (q < remainder) {
            start_row = q * (partSize + 1);
            end_row   = start_row + (partSize + 1);
        }
        else {
            start_row = q * partSize + remainder;
            end_row   = start_row + partSize;
        }

        for (int i = start_row; i < end_row; i++)
            lu_row_offsets[i] += new_nnz;

        new_nnz += lu_column_indices[q].size();

        lu_column_indices[0].insert(lu_column_indices[0].end(), lu_column_indices[q].begin(), lu_column_indices[q].end());
        lu_values[0].insert(lu_values[0].end(), lu_values[q].begin(), lu_values[q].end());
    }
    lu_row_offsets[m_n - partSize] = new_nnz;

    for (int q = numPartitions - 2; q >= 1; q--) {
        int start_row = 0, end_row = 0;

        if (q < remainder) {
            start_row = q * (partSize + 1);
            end_row   = start_row + (partSize + 1);
        }
        else {
            start_row = q * partSize + remainder;
            end_row   = start_row + partSize;
        }

        for (int i = end_row - 1; i >= start_row; i--)
            ul_row_offsets[i] += new_nnz_ul;

        new_nnz_ul += ul_column_indices[q - 1].size();

        ul_column_indices[numPartitions - 2].insert(ul_column_indices[numPartitions - 2].end(), ul_column_indices[q - 1].begin(), ul_column_indices[q - 1].end());
        ul_values[numPartitions - 2].insert(ul_values[numPartitions - 2].end(), ul_values[q - 1].begin(), ul_values[q - 1].end());
    }
    ul_row_offsets[remainder == 0 ? (partSize-1) : partSize] = new_nnz_ul;

    {
        IntVectorH& final_lu_column_indices = lu_column_indices[0];
        PrecVectorH& final_lu_values        = lu_values[0];

        Acsrh.resize(m_n, m_n, new_nnz);
        cusp::blas::fill(Acsrh.row_offsets, 0);

        IntVectorH  tmp_row_indices(new_nnz);
        IntVectorH  tmp_column_indices(new_nnz);
        PrecVectorH tmp_values(new_nnz);

        for (int i = 0; i < new_nnz; i++)
            Acsrh.row_offsets[final_lu_column_indices[i]] ++;

        thrust::exclusive_scan(Acsrh.row_offsets.begin(), Acsrh.row_offsets.end(), Acsrh.row_offsets.begin());

        for (int i = 0; i < m_n - partSize; i++) {
            int start_idx = lu_row_offsets[i];
            int end_idx   = lu_row_offsets[i+1];

            for (int l = start_idx; l < end_idx; l++) {
                int cur_k = final_lu_column_indices[l];
                int idx = Acsrh.row_offsets[cur_k];
                Acsrh.row_offsets[cur_k] ++;
                tmp_row_indices[idx] = i;
                tmp_column_indices[idx] = cur_k; 
                tmp_values[idx] = final_lu_values[l]; 
            }
        }

        cusp::blas::fill(Acsrh.row_offsets, 0);
        for (int i = 0; i < new_nnz; i++)
            Acsrh.row_offsets[tmp_row_indices[i]] ++;

        thrust::inclusive_scan(Acsrh.row_offsets.begin(), Acsrh.row_offsets.end(), Acsrh.row_offsets.begin());

        for (int i = new_nnz - 1; i >= 0; i--) {
            int idx = --Acsrh.row_offsets[tmp_row_indices[i]];
            Acsrh.column_indices[idx] = tmp_column_indices[i];
            Acsrh.values[idx] = tmp_values[i];
        }
        // cusp::blas::copy(lu_row_offsets, Acsrh.row_offsets);
    }

    {
        IntVectorH& final_ul_column_indices = ul_column_indices[numPartitions - 2];
        PrecVectorH& final_ul_values        = ul_values[numPartitions - 2];

        Acsrh2.resize(m_n, m_n, new_nnz_ul);
        cusp::blas::fill(Acsrh2.row_offsets, 0);

        IntVectorH  tmp_row_indices(new_nnz_ul);
        IntVectorH  tmp_column_indices(new_nnz_ul);
        PrecVectorH tmp_values(new_nnz_ul);

        for (int i = 0; i < new_nnz_ul; i++)
            Acsrh2.row_offsets[final_ul_column_indices[i]] ++;

        thrust::exclusive_scan(Acsrh2.row_offsets.begin(), Acsrh2.row_offsets.end(), Acsrh2.row_offsets.begin());

        for (int i = (remainder == 0 ? partSize : (partSize + 1)); i < m_n; i++) {
            int start_idx = ul_row_offsets[i];
            int end_idx   = ul_row_offsets[i-1];

            for (int l = start_idx; l < end_idx; l++) {
                int cur_k = final_ul_column_indices[l];
                int idx = Acsrh2.row_offsets[cur_k];
                Acsrh2.row_offsets[cur_k] ++;
                tmp_row_indices[idx] = i;
                tmp_column_indices[idx] = cur_k; 
                tmp_values[idx] = final_ul_values[l]; 
            }
        }

        cusp::blas::fill(Acsrh2.row_offsets, 0);
        for (int i = 0; i < new_nnz_ul; i++)
            Acsrh2.row_offsets[tmp_row_indices[i]] ++;

        thrust::inclusive_scan(Acsrh2.row_offsets.begin(), Acsrh2.row_offsets.end(), Acsrh2.row_offsets.begin());

        for (int i = new_nnz_ul - 1; i >= 0; i--) {
            int idx = --Acsrh2.row_offsets[tmp_row_indices[i]];
            Acsrh2.column_indices[idx] = tmp_column_indices[i];
            Acsrh2.values[idx] = tmp_values[i];
        }
        // cusp::blas::copy(lu_row_offsets, Acsrh.row_offsets);
    }
}

/*! \brief This function does ILU0 to the provided CSR matrix.
 */
template <typename PrecVector>
void
Precond<PrecVector>::ILU0(PrecMatrixCsrH& Acsrh)
{
    IntVectorH&  row_offsets    = m_Acsrh.row_offsets;
    IntVectorH&  column_indices = m_Acsrh.column_indices;
    PrecVectorH& values         = m_Acsrh.values;

    for (int i = 1; i < m_n; i++) {
        int start_idx = row_offsets[i];
        int end_idx = row_offsets[i+1];

        for (int l = start_idx; l < end_idx; l++) {
            int cur_k = column_indices[l];
            if (cur_k >= i)
                break;

            int start_k_idx = row_offsets[cur_k];
            int end_k_idx = row_offsets[cur_k+1];

            int l2;
            PrecValueType val_i_k = 0.0;

            for (l2 = start_k_idx; l2 < end_k_idx; l2++) {
                int pivot = column_indices[l2];
                if (pivot > cur_k) 
                    throw system_error(system_error::Zero_pivoting, "Found a pivot equal to zero (ilu).");
                else if (pivot == cur_k) {
                    val_i_k = (values[l] /= values[l2]);
                    break;
                }
            }

            if (l2 >= end_k_idx)
                throw system_error(system_error::Zero_pivoting, "Found a pivot equal to zero (ilu).");

            int l3 = l + 1;
            for (l2++; l2 < end_k_idx; l2++) {
                int tar_j = column_indices[l2];
                for (; l3 < end_idx; l3++) {
                    int tmp_j = column_indices[l3];
                    if (tmp_j > tar_j) break;
                    else if (tmp_j == tar_j) {
                        values[l3] -= values[l2] * val_i_k;
                        l3 ++;
                        break;
                    }
                }
                if (l3 >= end_idx) break;
            }
        }
    } 
}

/*! \brief This function does incomplete LU with pivoting 
 * to the provided CSR matrix.
 *
 * The integer p specifies the filling-in factor, tau
 * indicates the threshold of drop-off and perm_tol indicates
 * the threshold of candidate of pivoting. perm and reordering
 * indicate the permutation and inverse permutation due
 * to pivoting.
 */
template <typename PrecVector>
void
Precond<PrecVector>::ILUTP(PrecMatrixCsrH&    Acsrh,
                           int                p,
                           PrecValueType      tau,
                           PrecValueType      perm_tol,
                           IntVectorH&        perm,
                           IntVectorH&        reordering)
{
    IntVectorH    lu_row_offsets(m_n + 1, 0);
    IntVectorH    lu_column_indices;
    PrecVectorH   lu_values;
    PrecVectorH   wvector(m_n, 0);
    IntVectorH    in_wvector(m_n, 0);

    perm.resize(m_n);
    reordering.resize(m_n);
    thrust::sequence(perm.begin(), perm.end());
    cusp::blas::copy(perm, reordering);

    IntVectorH pivot_positions(m_n, -1);
    
    {
        lu_row_offsets[0] = Acsrh.row_offsets[0];
        lu_row_offsets[1] = Acsrh.row_offsets[1];
        int start_idx = Acsrh.row_offsets[0];
        int end_idx = Acsrh.row_offsets[1];

        lu_column_indices.resize(end_idx - start_idx);
        lu_values.resize(end_idx - start_idx);

        thrust::copy(Acsrh.column_indices.begin() + start_idx, Acsrh.column_indices.begin() + end_idx, lu_column_indices.begin());
        thrust::copy(Acsrh.values.begin() + start_idx, Acsrh.values.begin() + end_idx, lu_values.begin());

        int l;
        int pivot_col = -1;
        int pivot_pos = -1;
        PrecValueType max_pivot_val = 0.0;
        PrecValueType cur_pivot_val = 0.0;
        for (l = start_idx; l < end_idx; l++) {
            PrecValueType tmp_val = Acsrh.values[l];
            int           tmp_col = Acsrh.column_indices[l];

            if (tmp_col == 0) {
                cur_pivot_val = tmp_val;
                max_pivot_val = fabs(tmp_val);
                pivot_col = tmp_col;
                pivot_pos = l;
                break;
            }
        }

        for (l = start_idx; l < end_idx; l++) {
            PrecValueType tmp_val = Acsrh.values[l];
            int           tmp_col = Acsrh.column_indices[l];
            if (tmp_col == 0)
                continue;
            else {
                if (fabs(tmp_val) * perm_tol > fabs(cur_pivot_val)) {
                    if (fabs(tmp_val) > max_pivot_val) {
                        pivot_col = tmp_col;
                        pivot_pos = l;
                        max_pivot_val = fabs(tmp_val);
                    }
                }
            }
        }

        if (pivot_col < 0)
            throw system_error(system_error::Zero_pivoting, "Found a pivot equal to zero (ilu).");

        m_pivots[0] = Acsrh.values[pivot_pos];
        pivot_positions[0] = pivot_pos;

        if (pivot_col != 0) {
            perm[0]               = pivot_col;
            perm[pivot_col]       = 0;
            reordering[0]         = pivot_col;
            reordering[pivot_col] = 0;
        }
    }

    IntVectorH w_nonzeros(m_n);
    int wvec_size;

    IntVectorH l_columns(m_n), u_columns(m_n);
    PrecVectorH l_values(m_n), u_values(m_n);
    int l_size, u_size;

    for (int i = 1; i < m_n; i++) {
        int start_idx = Acsrh.row_offsets[i];
        int end_idx = Acsrh.row_offsets[i+1];

        if (end_idx <= start_idx)
            throw system_error(system_error::Matrix_singular, "Singular matrix found");

        int nl = 0, nu = 0;

        PrecValueType tau_i = (PrecValueType)0;

        wvec_size = end_idx - start_idx;

        std::priority_queue<int, std::vector<int>, std::greater<int> > pq;
        for (int l = start_idx; l < end_idx; l++) {
            PrecValueType tmp_val = Acsrh.values[l];
            int cur_k = Acsrh.column_indices[l];
            wvector[cur_k] = tmp_val;
            in_wvector[cur_k] = l - start_idx + 1;
            tau_i += tmp_val * tmp_val;
            w_nonzeros[l - start_idx] = cur_k;
            int permed_k = perm[cur_k];
            if (permed_k < i) {
                pq.push(permed_k);
                nl ++;
            } else  if (permed_k > i)
                nu ++;
        }
        tau_i = sqrt(tau_i) / (end_idx - start_idx) * tau;

        while (!pq.empty()) {
            int permed_k = pq.top();
            pq.pop();
            int cur_k = reordering[permed_k];

            int start_k_idx = lu_row_offsets[permed_k];
            int end_k_idx = lu_row_offsets[permed_k+1];

            int l2 = pivot_positions[permed_k];
            PrecValueType val_i_k = (PrecValueType)0;

            val_i_k = (wvector[cur_k] /= lu_values[l2]);

            // Applying drop-off to w[cur_k]
            if (fabs(val_i_k) < tau_i) {
                in_wvector[cur_k] = 0;
                continue;
            }

            for (l2 = start_k_idx; l2 < end_k_idx; l2++) {
                int tar_j = lu_column_indices[l2];
                int permed_j = perm[tar_j];

                if (permed_j <= permed_k)
                    continue;

                wvector[tar_j] -= val_i_k * lu_values[l2];

                if(!in_wvector[tar_j]) {
                    w_nonzeros[wvec_size++] = tar_j;
                    in_wvector[tar_j] = wvec_size;
                    if (permed_j < i)
                        pq.push(permed_j);
                }
            }
        } // end while

        // Apply drop-off to wvector
        {
            int pivot_col = -1;
            int pivot_pos = -1;
            PrecValueType max_pivot_val = 0.0;
            PrecValueType cur_pivot_val = 0.0;
            for (int w_it = 0; w_it < wvec_size; w_it++) {
                int cur_k = w_nonzeros[w_it];
                if (!in_wvector[cur_k])
                    continue;
                int permed_k = perm[cur_k];

                if (permed_k == i) {
                    PrecValueType tmp_val = wvector[cur_k];
                    pivot_col     = i;
                    max_pivot_val = fabs(tmp_val);
                    cur_pivot_val = tmp_val;
                    pivot_pos     = w_it;
                    break;
                }
            }

            for (int w_it = 0; w_it < wvec_size; w_it++) {
                int cur_k = w_nonzeros[w_it];
                if (!in_wvector[cur_k])
                    continue;
                int permed_k = perm[cur_k];
                if (permed_k <= i)
                    continue;

                PrecValueType tmp_val = wvector[cur_k];

                if (fabs(tmp_val) * perm_tol > fabs(cur_pivot_val)) {
                    if (fabs(tmp_val) > max_pivot_val) {
                        pivot_col = permed_k;
                        max_pivot_val = fabs(tmp_val);
                        pivot_pos   = w_it;
                    }
                }
            }

            if (pivot_col >= 0) {
                m_pivots[i] = wvector[w_nonzeros[pivot_pos]];

                if (pivot_col != i) {
                    int cur_col = reordering[pivot_col];
                    int cur_i   = reordering[i];
                    perm[cur_col] = i;
                    perm[cur_i]   = pivot_col;
                    reordering[i] = cur_col;
                    reordering[pivot_col] = cur_i;
                }
            } else
                throw system_error(system_error::Zero_pivoting, "Found a pivot equal to zero (ILUTP).");

            l_size = u_size = 0;
            for (int w_it = 0; w_it < wvec_size; w_it++) {
                int cur_k = w_nonzeros[w_it];
                if (!in_wvector[cur_k])
                    continue;
                PrecValueType tmp_val = wvector[cur_k];
                int permed_k = perm[cur_k];

                if (permed_k == i) {
                    m_pivots[i] = (pivot_col < 0 ? perm_tol : tmp_val);
                    continue;
                }

                if (fabs(tmp_val) < tau_i)
                    continue;

                if (permed_k < i) {
                    l_columns[l_size] = cur_k;
                    l_values[l_size]  = tmp_val;
                    l_size++;
                } else {
                    u_columns[u_size] = cur_k;
                    u_values[u_size]  = tmp_val;
                    u_size++;
                }
            }

            // Clear the content of wvector and in_wvector for usage of next iteration
            for (int w_it = 0; w_it < wvec_size; w_it++) {
                int cur_k = w_nonzeros[w_it];
                wvector[cur_k] = (PrecValueType)0;
                in_wvector[cur_k] = 0;
            }

            if (l_size > p + nl) {
                findPthMax(l_columns.begin(), l_columns.begin() + l_size, 
                        l_values.begin(),  l_values.begin() + l_size,
                        p + nl);
                l_size = p + nl;
            }

            if (u_size > p + nu) {
                findPthMax(u_columns.begin(), u_columns.begin() + u_size, 
                        u_values.begin(),  u_values.begin() + u_size,
                        p + nu);
                u_size = p + nu;
            }

            lu_column_indices.insert(lu_column_indices.end(), l_columns.begin(), l_columns.begin() + l_size);
            lu_values.insert(lu_values.end(), l_values.begin(), l_values.begin() + l_size);

            lu_column_indices.push_back(reordering[i]);
            lu_values.push_back(m_pivots[i]);
            pivot_positions[i] = lu_column_indices.size() - 1;

            lu_column_indices.insert(lu_column_indices.end(), u_columns.begin(), u_columns.begin() + u_size);
            lu_values.insert(lu_values.end(), u_values.begin(), u_values.begin() + u_size);

            lu_row_offsets[i+1] = lu_column_indices.size();
        }
    } // end for

    int new_nnz = lu_column_indices.size();
    Acsrh.resize(m_n, m_n, new_nnz);
    cusp::blas::fill(Acsrh.row_offsets, 0);

    IntVectorH  tmp_row_indices(new_nnz);
    IntVectorH  tmp_column_indices(new_nnz);
    PrecVectorH tmp_values(new_nnz);

    for (int i = 0; i < new_nnz; i++) {
        int tmp_col = (lu_column_indices[i] = perm[lu_column_indices[i]]);
        Acsrh.row_offsets[tmp_col] ++;
    }

    thrust::exclusive_scan(Acsrh.row_offsets.begin(), Acsrh.row_offsets.end(), Acsrh.row_offsets.begin());

    for (int i = 0; i < m_n; i++) {
        int start_idx = lu_row_offsets[i];
        int end_idx   = lu_row_offsets[i+1];

        for (int l = start_idx; l < end_idx; l++) {
            int cur_k = lu_column_indices[l];
            int idx = Acsrh.row_offsets[cur_k];
            Acsrh.row_offsets[cur_k] ++;
            tmp_row_indices[idx] = i;
            tmp_column_indices[idx] = cur_k; 
            tmp_values[idx] = lu_values[l]; 
        }
    }

    cusp::blas::fill(Acsrh.row_offsets, 0);
    for (int i = 0; i < new_nnz; i++)
        Acsrh.row_offsets[tmp_row_indices[i]] ++;

    thrust::inclusive_scan(Acsrh.row_offsets.begin(), Acsrh.row_offsets.end(), Acsrh.row_offsets.begin());

    for (int i = new_nnz - 1; i >= 0; i--) {
        int idx = --Acsrh.row_offsets[tmp_row_indices[i]];
        Acsrh.column_indices[idx] = tmp_column_indices[i];
        Acsrh.values[idx] = tmp_values[i];
    }

    cusp::blas::copy(lu_row_offsets, Acsrh.row_offsets);
}

/** This function does sparse factorization to the provided
 * CSR matrix.
 */
template <typename PrecVector>
void
Precond<PrecVector>::sparseFactorization()
{
    m_pivots.resize(m_n);
    m_pivots_ul.resize(m_n);

    if (m_variableBandwidth || m_numPartitions == 1) {

        IntVectorH pivotPerm, pivotReordering;

        // Do ILU here, now only ILU0 is implemented
        if (m_ilu_level == 0)
            ILU0(m_Acsrh);
        else {
            if (m_safeFactorization)
                ILUTP(m_Acsrh, m_ilu_level, m_tolerance, (PrecValueType)0.1, pivotPerm, pivotReordering);
            else
                ILUT(m_Acsrh, m_ilu_level, m_tolerance);
        }

        for (int i = 0; i < m_n; i++) {
            int start_idx = m_Acsrh.row_offsets[i];
            int end_idx = m_Acsrh.row_offsets[i+1];

            PrecValueType pivot_val = (PrecValueType)(0);
            for (int l = start_idx; l < end_idx; l++) {
                int cur_k = m_Acsrh.column_indices[l];
                if (cur_k == i)
                    m_pivots[i] = pivot_val = m_Acsrh.values[l];
                else if (cur_k > i)
                    m_Acsrh.values[l] /= pivot_val;
            }
        }

        if (m_ilu_level > 0 && m_safeFactorization) {
            IntVector buffer = pivotReordering, buffer2(m_n);
            combinePermutation(buffer, m_optReordering, buffer2);
            m_optReordering = buffer2;
        }
    } else {
        ILUULT(m_Acsrh, m_Acsrh_ul, m_ilu_level, m_tolerance);

        int partSize  = m_n / m_numPartitions;
        int remainder = m_n % m_numPartitions;

        for (int i = 0; i < m_n - partSize; i++) {
            int start_idx = m_Acsrh.row_offsets[i];
            int end_idx = m_Acsrh.row_offsets[i+1];

            PrecValueType pivot_val = (PrecValueType)(0);
            for (int l = start_idx; l < end_idx; l++) {
                int cur_k = m_Acsrh.column_indices[l];
                if (cur_k == i)
                    pivot_val = m_Acsrh.values[l];
                else if (cur_k > i)
                    m_Acsrh.values[l] /= pivot_val;
            }
        }

        for (int i = (remainder == 0 ? partSize : (partSize + 1)); i < m_n; i++) {
            int start_idx = m_Acsrh_ul.row_offsets[i];
            int end_idx = m_Acsrh_ul.row_offsets[i+1];

            PrecValueType pivot_val = (PrecValueType)(0);
            for (int l = end_idx - 1; l >= start_idx; l--) {
                int cur_k = m_Acsrh_ul.column_indices[l];
                if (cur_k == i)
                    pivot_val = m_Acsrh_ul.values[l];
                else if (cur_k < i)
                    m_Acsrh_ul.values[l] /= pivot_val;
            }
        }

        thrust::copy(m_pivots_ul.begin() + (m_n - partSize), m_pivots_ul.end(), m_pivots.begin() + (m_n - partSize));

        for (int i = m_n - partSize; i < m_n; i++) {
            int start_idx = m_Acsrh_ul.row_offsets[i];
            int end_idx   = m_Acsrh_ul.row_offsets[i + 1];
            m_Acsrh.column_indices.insert(m_Acsrh.column_indices.end(), m_Acsrh_ul.column_indices.begin() + start_idx, m_Acsrh_ul.column_indices.begin() + end_idx);
            m_Acsrh.values.insert(m_Acsrh.values.end(), m_Acsrh_ul.values.begin() + start_idx, m_Acsrh_ul.values.begin() + end_idx);

            m_Acsrh.row_offsets[i+1] = m_Acsrh.column_indices.size();
        }
        m_Acsrh.num_entries = m_Acsrh.column_indices.size();

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
        // partBandedLU_var();
        if (m_saveMem)
            partBlockedBandedCholesky_var();
        else
            partBlockedBandedLU_var();
    } else {
        // Constant bandwidth method.
        if (m_numPartitions > 1) {
            // partBandedLU_const();
            partBlockedBandedLU_const();
        }
        else {
            if (m_saveMem)
                partBlockedBandedCholesky_one();
            else
                partBlockedBandedLU_one();
        }
    }
}

template <typename PrecVector>
void
Precond<PrecVector>::partBandedLU_one()
{
    // As the name implies, this function can only be called if we arte using a single
    // partition. In this case, the entire banded matrix m_B is LU factorized.

    PrecValueType* dB = thrust::raw_pointer_cast(&m_B[0]);

    if (m_ks_col_host.size() != m_n)
        m_ks_col_host.resize(m_n, m_k);

    if (m_ks_row_host.size() != m_n)
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
                if (st_row == 0) {
                    if (m_safeFactorization)
                        device::bandLU_critical_div_onePart_safe_general<PrecValueType><<<1, 512>>>(dB, st_row, m_k, threadsNum);
                    else
                        device::bandLU_critical_div_onePart_general<PrecValueType><<<threadsNum/512+1, 512>>>(dB, st_row, m_k, threadsNum);
                }

                if (m_safeFactorization)
                    device::bandLU_critical_sub_div_onePart_safe_general<PrecValueType><<<blockX, 512>>>(dB, st_row, m_k, threadsNum, m_ks_col_host[st_row + 1]);
                else
                    device::bandLU_critical_sub_div_onePart_general<PrecValueType><<<blockX, 512>>>(dB, st_row, m_k, threadsNum, m_ks_col_host[st_row + 1]);
            } else {
                if (st_row == 0) {
                    if (m_safeFactorization)
                        device::bandLU_critical_div_onePart_safe<PrecValueType><<<1, threadsNum>>>(dB, st_row, m_k);
                    else
                        device::bandLU_critical_div_onePart<PrecValueType><<<1, threadsNum>>>(dB, st_row, m_k);
                }

                if (m_safeFactorization)
                    device::bandLU_critical_sub_div_onePart_safe<PrecValueType><<<blockX, threadsNum>>>(dB, st_row, m_k, m_ks_col_host[st_row+1]);
                else
                    device::bandLU_critical_sub_div_onePart<PrecValueType><<<blockX, threadsNum>>>(dB, st_row, m_k, m_ks_col_host[st_row + 1]);
            }
        }
    } else if (m_k > 27) {
        if (m_safeFactorization)
            device::bandLU_g32_safe<PrecValueType><<<1, 512>>>(dB, m_k, m_n, 0, false);
        else
            device::bandLU_g32<PrecValueType><<<1, 512>>>(dB, m_k, m_n, 0);
    } else {
        if (m_safeFactorization)
            device::bandLU_safe<PrecValueType><<<1,  m_k * m_k>>>(dB, m_k, m_n, 0, false);
        else
            device::bandLU<PrecValueType><<<1,  m_k * m_k>>>(dB, m_k, m_n, 0);
            ////device::swBandLU<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
    }


    if (m_safeFactorization)
        device::boostLastPivot<PrecValueType><<<1, 1>>>(dB, m_n, m_k, m_n, 0);


    // If not using safe factorization, check the factorized banded matrix for any
    // zeros on its diagonal (this means a zero pivot).
    if (!m_safeFactorization && hasZeroPivots(m_B.begin(), m_B.end(), m_k, 2 * m_k + 1, (PrecValueType) BURST_VALUE))
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
Precond<PrecVector>::partBlockedBandedLU_one()
{
    PrecValueType* dB = thrust::raw_pointer_cast(&m_B[0]);

    if (m_ks_col_host.size() != m_n) {
        m_ks_col_host.resize(m_n, m_k);

        for (int i = m_n - 1; i >= m_n - m_k; i--)
            m_ks_col_host[i] = m_n - 1 - i;
    }

    if (m_ks_row_host.size() != m_n) {
        m_ks_row_host.resize(m_n, m_k);

        for (int i = m_n - 1; i >= m_n - m_k; i--)
            m_ks_row_host[i] = m_n - 1 - i;
    }

    if(m_k >= CRITICAL_THRESHOLD) {
        int threadsNum = 0;

        const int BLOCK_FACTOR = 8;

        IntVector ks_col = m_ks_col_host;
        int *ks_col_ptr = thrust::raw_pointer_cast(&ks_col[0]);
        for (int st_row = 0; st_row < m_n-1; st_row += BLOCK_FACTOR) {
            int last_row = st_row + BLOCK_FACTOR;
            if (last_row > m_n)
                last_row = m_n;

            int col_max = thrust::reduce(m_ks_col_host.begin() + st_row, m_ks_col_host.begin() + last_row, 0, thrust::maximum<int>());

            threadsNum = col_max * (last_row - st_row - 1);

            int blockX = 0;
            for (int i = st_row; i < last_row; i++)
                if (blockX < i + m_ks_row_host[i])
                    blockX = i + m_ks_row_host[i];

            blockX -= st_row + BLOCK_FACTOR - 1;

            if (m_safeFactorization) {
                if (threadsNum > 1024)
                    device::blockedBandLU_critical_phase1_safe<PrecValueType><<<1, 512>>>(dB, st_row, m_k, ks_col_ptr, last_row - st_row);
                else
                    device::blockedBandLU_critical_phase1_safe<PrecValueType><<<1, threadsNum>>>(dB, st_row, m_k, ks_col_ptr, last_row - st_row);
            } else {
                if (threadsNum > 1024)
                    device::blockedBandLU_critical_phase1<PrecValueType><<<1, 512>>>(dB, st_row, m_k, ks_col_ptr, last_row - st_row);
                else
                    device::blockedBandLU_critical_phase1<PrecValueType><<<1, threadsNum>>>(dB, st_row, m_k, ks_col_ptr, last_row - st_row);
            }

            if (m_n == last_row) break;

            if (blockX <= 0) continue;

            device::blockedBandLU_critical_phase2<PrecValueType><<<blockX, BLOCK_FACTOR, BLOCK_FACTOR * sizeof(PrecValueType)>>>(dB, st_row, m_k, BLOCK_FACTOR);


            threadsNum = col_max;
            int row_max = thrust::reduce(m_ks_row_host.begin() + st_row, m_ks_row_host.begin() + last_row, 0, thrust::maximum<int>());

            if (threadsNum > 1024)
                device::blockedBandLU_critical_phase3<PrecValueType><<<blockX, 512, sizeof(PrecValueType) * BLOCK_FACTOR>>>(dB, st_row, m_k, threadsNum, row_max, BLOCK_FACTOR);
            else
                device::blockedBandLU_critical_phase3<PrecValueType><<<blockX, threadsNum, sizeof(PrecValueType) * BLOCK_FACTOR>>>(dB, st_row, m_k, threadsNum, row_max, BLOCK_FACTOR);

        }
    } else if (m_k > 27) {
        if (m_safeFactorization)
            device::bandLU_g32_safe<PrecValueType><<<1, 512>>>(dB, m_k, m_n, 0, false);
        else
            device::bandLU_g32<PrecValueType><<<1, 512>>>(dB, m_k, m_n, 0);
    } else {
        if (m_safeFactorization)
            device::bandLU_safe<PrecValueType><<<1,  m_k * m_k>>>(dB, m_k, m_n, 0, false);
        else
            device::bandLU<PrecValueType><<<1,  m_k * m_k>>>(dB, m_k, m_n, 0);
            ////device::swBandLU<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
    }

    if (m_safeFactorization)
        device::boostLastPivot<PrecValueType><<<1, 1>>>(dB, m_n, m_k, m_n, 0);


    // If not using safe factorization, check the factorized banded matrix for any
    // zeros on its diagonal (this means a zero pivot).
    if (!m_safeFactorization && hasZeroPivots(m_B.begin(), m_B.end(), m_k, 2 * m_k + 1, (PrecValueType) BURST_VALUE))
        throw system_error(system_error::Zero_pivoting, "Found a pivot equal to zero (partBlockedBandedLU_one).");


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
Precond<PrecVector>::partBlockedBandedCholesky_one()
{
    PrecValueType* dB = thrust::raw_pointer_cast(&m_B[0]);

    if (m_ks_col_host.size() != m_n) {
        m_ks_col_host.resize(m_n, m_k);

        for (int i = m_n - 1; i >= m_n - m_k; i--)
            m_ks_col_host[i] = m_n - 1 - i;
    }

    if (m_ks_row_host.size() != m_n) {
        m_ks_row_host.resize(m_n, m_k);

        for (int i = m_n - 1; i >= m_n - m_k; i--)
            m_ks_row_host[i] = m_n - 1 - i;
    }

    if(m_k >= CRITICAL_THRESHOLD) {
        int threadsNum = 0;

        const int BLOCK_FACTOR = 8;

        IntVector ks_col = m_ks_col_host;
        int *ks_col_ptr = thrust::raw_pointer_cast(&ks_col[0]);
        for (int st_row = 0; st_row < m_n-1; st_row += BLOCK_FACTOR) {
            int last_row = st_row + BLOCK_FACTOR;
            if (last_row > m_n)
                last_row = m_n;

            int col_max = thrust::reduce(m_ks_col_host.begin() + st_row, m_ks_col_host.begin() + last_row, 0, thrust::maximum<int>());

            threadsNum = col_max * (last_row - st_row - 1);

            int blockX = 0;
            for (int i = st_row; i < last_row; i++)
                if (blockX < i + m_ks_row_host[i])
                    blockX = i + m_ks_row_host[i];

            blockX -= st_row + BLOCK_FACTOR - 1;

            device::blockedCholesky_critical_phase1_safe<PrecValueType><<<1, 512>>>(dB, st_row, m_k, ks_col_ptr, last_row - st_row);

            if (m_n == last_row) break;

            if (blockX <= 0) continue;

            threadsNum = col_max;

            device::blockedCholesky_critical_phase2<PrecValueType><<<blockX, 512, sizeof(PrecValueType) * BLOCK_FACTOR>>>(dB, st_row, m_k, threadsNum, BLOCK_FACTOR);
        }

    } else if (m_k > 27) {
        device::bandLU_g32_safe<PrecValueType><<<1, 512>>>(dB, m_k, m_n, 0, true);
    } else {
        device::bandLU_safe<PrecValueType><<<1,  m_k * m_k>>>(dB, m_k, m_n, 0, true);
            ////device::swBandLU<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
    }
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
            device::bandLU_g32_safe<PrecValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder, false);
        else
            device::bandLU_g32<PrecValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder);
    } else {
        if (m_safeFactorization)
            device::bandLU_safe<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder, false);
        else
            device::bandLU<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
            ////device::swBandLU<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
    }


    // If not using safe factorization, check the factorized banded matrix for any
    // zeros on its diagonal (this means a zero pivot). Note that we must only check
    // the diagonal blocks corresponding to the partitions for which LU was applied.
    if (!m_safeFactorization && hasZeroPivots(m_B.begin(), m_B.begin() + n_eff * (2*m_k+1), m_k, 2 * m_k + 1, (PrecValueType) BURST_VALUE))
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
Precond<PrecVector>::partBlockedBandedLU_const()
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
        int final_partition_size = (remainder == 0 ? partSize : (partSize + 1));
        int threadsNum = m_k;
        if (threadsNum > 512)
            threadsNum = 512;

        const int BLOCK_FACTOR = 8;
        for (int st_row = 0; st_row < final_partition_size; st_row += BLOCK_FACTOR) {
            int last_row = st_row + BLOCK_FACTOR;
            if (last_row > final_partition_size)
                last_row = final_partition_size;
            int block_num = last_row - st_row;

            device::blockedBandLU_critical_const_phase1_safe<PrecValueType> <<< numPart_eff, threadsNum>>>(dB, st_row, m_k, block_num, partSize, remainder);

            if (last_row == final_partition_size)
                break;

            dim3 grids(m_k, numPart_eff);

            device::blockedBandLU_critical_const_phase2<PrecValueType> <<<grids, BLOCK_FACTOR, BLOCK_FACTOR * sizeof(PrecValueType)>>> (dB, st_row, m_k, BLOCK_FACTOR, partSize, remainder);

            device::blockedBandLU_critical_const_phase3<PrecValueType> <<<grids, threadsNum, BLOCK_FACTOR * sizeof(PrecValueType)>>> (dB, st_row, m_k, BLOCK_FACTOR, partSize, remainder);
        }
    } else if (m_k > 27) {
        if (m_safeFactorization)
            device::bandLU_g32_safe<PrecValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder, false);
        else
            device::bandLU_g32<PrecValueType><<<numPart_eff, 512>>>(dB, m_k, partSize, remainder);
    } else {
        if (m_safeFactorization)
            device::bandLU_safe<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder, false);
        else
            device::bandLU<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
            ////device::swBandLU<PrecValueType><<<numPart_eff,  m_k * m_k>>>(dB, m_k, partSize, remainder);
    }


    // If not using safe factorization, check the factorized banded matrix for any
    // zeros on its diagonal (this means a zero pivot). Note that we must only check
    // the diagonal blocks corresponding to the partitions for which LU was applied.
    if (!m_safeFactorization && hasZeroPivots(m_B.begin(), m_B.begin() + n_eff * (2*m_k+1), m_k, 2 * m_k + 1, (PrecValueType) BURST_VALUE))
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

        IntVector ks_col = m_ks_col_host;
        IntVector ks_row = m_ks_row_host;
        int *ks_col_ptr = thrust::raw_pointer_cast(&ks_col[0]);
        int *ks_row_ptr = thrust::raw_pointer_cast(&ks_row[0]);

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
                // last = 0;
                int corres_row = st_row;
                for (int i = 0; i < remainder; i++) {
                    if (blockY < m_ks_row_host[corres_row])
                        blockY = m_ks_row_host[corres_row];
                    //if (last < m_ks_col_host[corres_row])
                        //last = m_ks_col_host[corres_row];
                    corres_row += partSize + 1;
                }
                corres_row --;
                for (int i = remainder; i < m_numPartitions; i++) {
                    if (blockY < m_ks_row_host[corres_row])
                        blockY = m_ks_row_host[corres_row];
                    //if (last < m_ks_col_host[corres_row])
                        //last = m_ks_col_host[corres_row];
                    corres_row += partSize;
                }

                if (st_row == 1) {
                    if (m_safeFactorization)
                        device::var::bandLU_critical_div_safe_general<PrecValueType><<<m_numPartitions, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder);
                    else
                        device::var::bandLU_critical_div_general<PrecValueType><<<m_numPartitions, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder);
                }

                dim3 tmpGrids(blockY, m_numPartitions);
                if (m_safeFactorization)
                    device::var::bandLU_critical_sub_div_safe_general<PrecValueType><<<tmpGrids, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder, ks_col_ptr, ks_row_ptr);
                else
                    device::var::bandLU_critical_sub_div_general<PrecValueType><<<tmpGrids, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, partSize, remainder, ks_col_ptr, ks_row_ptr);
            }
        }
    } else if (tmp_k > 27){
        device::var::bandLU_g32_safe<PrecValueType><<<m_numPartitions, 512>>>(dB, p_ks, p_BOffsets, partSize, remainder, false);
    } else {
        device::var::bandLU_safe<PrecValueType><<<m_numPartitions,  tmp_k * tmp_k >>>(dB, p_ks, p_BOffsets, partSize, remainder, false);
    }


    if (m_safeFactorization)
        device::var::boostLastPivot<PrecValueType><<<m_numPartitions, 1>>>(dB, partSize, p_ks, p_BOffsets, partSize, remainder);


    // If not using safe factorization, check for zero pivots in the factorized banded
    // matrix, one partition at a time.
    if (!m_safeFactorization) {
        for (int i = 0; i < m_numPartitions; i++) {
            if (hasZeroPivots(m_B.begin() + m_BOffsets_host[i], m_B.begin() + m_BOffsets_host[i+1], m_ks_host[i], 2 * m_ks_host[i] + 1, (PrecValueType) BURST_VALUE))
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

template <typename PrecVector>
void
Precond<PrecVector>::partBlockedBandedLU_var()
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

    if(tmp_k >= CRITICAL_THRESHOLD) 
    {
        int final_partition_size = (remainder == 0 ? partSize : (partSize + 1));
        int threadsNum = adjustNumThreads(cusp::blas::nrm1(m_ks) / m_numPartitions);

        IntVector ks_col = m_ks_col_host;
        int *ks_col_ptr = thrust::raw_pointer_cast(&ks_col[0]);

        const int BLOCK_FACTOR = 8;
        for (int st_row = 0; st_row < final_partition_size; st_row += BLOCK_FACTOR) {
            int last_row = st_row + BLOCK_FACTOR;
            if (last_row > final_partition_size)
                last_row = final_partition_size;
            int block_num = last_row - st_row;

            device::var::blockedBandLU_critical_phase1_safe<PrecValueType> <<< m_numPartitions, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, ks_col_ptr, block_num, partSize, remainder, false);

            if (last_row == final_partition_size)
                break;

            dim3 grids(tmp_k, m_numPartitions);

            device::var::blockedBandLU_critical_phase2<PrecValueType> <<<grids, BLOCK_FACTOR, BLOCK_FACTOR * sizeof(PrecValueType)>>> (dB, st_row, p_ks, p_BOffsets, BLOCK_FACTOR, partSize, remainder);

            device::var::blockedBandLU_critical_phase3<PrecValueType> <<<grids, threadsNum, BLOCK_FACTOR * sizeof(PrecValueType)>>> (dB, st_row, p_ks, p_BOffsets, BLOCK_FACTOR, partSize, remainder, false);
        }
    } else if (tmp_k > 27)
        device::var::bandLU_g32_safe<PrecValueType><<<m_numPartitions, 512>>>(dB, p_ks, p_BOffsets, partSize, remainder, false);
    else
        device::var::bandLU_safe<PrecValueType><<<m_numPartitions,  tmp_k * tmp_k >>>(dB, p_ks, p_BOffsets, partSize, remainder, false);

    if (m_safeFactorization)
        device::var::boostLastPivot<PrecValueType><<<m_numPartitions, 1>>>(dB, partSize, p_ks, p_BOffsets, partSize, remainder);

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

template <typename PrecVector>
void
Precond<PrecVector>::partBlockedBandedCholesky_var()
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

    if(tmp_k >= CRITICAL_THRESHOLD) 
    {
        int final_partition_size = partSize + 1;
        int threadsNum = adjustNumThreads(cusp::blas::nrm1(m_ks) / m_numPartitions);

        IntVector ks_col = m_ks_col_host;
        int *ks_col_ptr = thrust::raw_pointer_cast(&ks_col[0]);

        const int BLOCK_FACTOR = 8;
        for (int st_row = 0; st_row < final_partition_size; st_row += BLOCK_FACTOR) {
            int last_row = st_row + BLOCK_FACTOR;
            if (last_row > final_partition_size)
                last_row = final_partition_size;
            int block_num = last_row - st_row;

            device::var::blockedBandLU_critical_phase1_safe<PrecValueType> <<< m_numPartitions, threadsNum>>>(dB, st_row, p_ks, p_BOffsets, ks_col_ptr, block_num, partSize, remainder, true);

            if (last_row == final_partition_size)
                break;

            dim3 grids(tmp_k, m_numPartitions);

            device::var::blockedBandLU_critical_phase3<PrecValueType> <<<grids, threadsNum, BLOCK_FACTOR * sizeof(PrecValueType)>>> (dB, st_row, p_ks, p_BOffsets, BLOCK_FACTOR, partSize, remainder, true);
        }
    } else if (tmp_k > 27)
        device::var::bandLU_g32_safe<PrecValueType><<<m_numPartitions, 512>>>(dB, p_ks, p_BOffsets, partSize, remainder, true);
    else
        device::var::bandLU_safe<PrecValueType><<<m_numPartitions,  tmp_k * tmp_k >>>(dB, p_ks, p_BOffsets, partSize, remainder, true);
    
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
    if (!m_safeFactorization && hasZeroPivots(B.begin() + (2 * m_k + 1) * n_first, B.end(), m_k, 2 * m_k + 1, (PrecValueType) BURST_VALUE))
        throw system_error(system_error::Zero_pivoting, "Found a pivot equal to zero (partBandedUL).");
}

template <typename PrecVector>
void 
Precond<PrecVector>::partBlockedBandedUL(PrecVector& B)
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

    if(m_k >= CRITICAL_THRESHOLD)
    {
        int threadsNum = m_k;
        int n_final = partSize + 1;
        if (threadsNum > 512)
            threadsNum = 512;

        const int BLOCK_FACTOR = 2;
        for (int st_row = n_final - 1; st_row >= 0; st_row -= BLOCK_FACTOR) {
            int last_row = st_row - BLOCK_FACTOR;
            if (last_row < 0)
                last_row = -1;
            int block_num = st_row - last_row;

            device::blockedBandUL_critical_const_phase1_safe<PrecValueType> <<< numPart_eff, threadsNum>>>(dB, st_row, m_k, block_num, partSize, remainder);

            if (last_row < 0)
                break;

            dim3 grids(m_k, numPart_eff);

            device::blockedBandUL_critical_const_phase2<PrecValueType> <<<grids, BLOCK_FACTOR, BLOCK_FACTOR * sizeof(PrecValueType)>>> (dB, st_row, m_k, BLOCK_FACTOR, partSize, remainder);

            device::blockedBandUL_critical_const_phase3<PrecValueType> <<<grids, threadsNum, BLOCK_FACTOR * sizeof(PrecValueType)>>> (dB, st_row, m_k, BLOCK_FACTOR, partSize, remainder);
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
    if (!m_safeFactorization && hasZeroPivots(B.begin() + (2 * m_k + 1) * n_first, B.end(), m_k, 2 * m_k + 1, (PrecValueType) BURST_VALUE))
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
        if (m_saveMem) {
            if (m_k > 1024)
                device::fwdElim_sol_forSPD<PrecValueType> <<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
            else
                device::fwdElim_sol_medium_forSPD<PrecValueType> <<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
        } else {
            if (m_k > 1024)
                device::forwardElimL_general<PrecValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
            else if (m_k > 32)
                device::forwardElimL_g32<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
            else
                device::forwardElimL<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
        }
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

    if (m_saveMem)
        if (tmp_k > 1024)
            device::var::fwdElimCholesky_sol<PrecValueType><<<m_numPartitions, 512>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
        else if (tmp_k > 32)
            device::var::fwdElimCholesky_sol_medium<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
        else
            device::var::fwdElimCholesky_sol_narrow<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
    else {
        if (tmp_k > 1024)
            device::var::fwdElim_sol<PrecValueType><<<m_numPartitions, 512>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
        else if (tmp_k > 32)
            device::var::fwdElim_sol_medium<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
        else
            device::var::fwdElim_sol_narrow<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
    }
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

            device::preBck_sol_divide<PrecValueType><<<grids, blockX>>>(m_n, m_k, p_B, p_v, partSize, remainder, m_saveMem);

            if (m_saveMem) {
                if (m_k > 1024)
                    device::bckElim_sol_forSPD<PrecValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
                else
                    device::bckElim_sol_medium_forSPD<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
            } else {
                if (m_k > 1024)
                    device::bckElim_sol<PrecValueType><<<m_numPartitions, 512>>>(m_n, m_k, p_B, p_v, partSize, remainder);
                else if (m_k > 32)
                    device::bckElim_sol_medium<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
                else
                    device::bckElim_sol_narrow<PrecValueType><<<m_numPartitions, m_k>>>(m_n, m_k, p_B, p_v, partSize, remainder);
            }
        }
    } else {
        strided_range<typename PrecVector::iterator> diag(m_B.begin() + m_k, m_B.end(), 2 * m_k + 1);
        thrust::transform(v.begin(), v.end(), diag.begin(), v.begin(), thrust::divides<PrecValueType>());
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
    device::var::preBck_sol_divide<PrecValueType><<<grids, blockX>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder, m_saveMem);

    if (m_saveMem) {
        if (tmp_k > 1024)
            device::var::bckElimCholesky_sol<PrecValueType><<<m_numPartitions, 512>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
        else if (tmp_k > 32) 
            device::var::bckElimCholesky_sol_medium<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
        else
            device::var::bckElimCholesky_sol_narrow<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
    }
    else {
        if (tmp_k > 1024)
            device::var::bckElim_sol<PrecValueType><<<m_numPartitions, 512>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
        else if (tmp_k > 32) 
            device::var::bckElim_sol_medium<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
        else
            device::var::bckElim_sol_narrow<PrecValueType><<<m_numPartitions, tmp_k>>>(m_n, p_ks, p_BOffsets, p_B, p_v, partSize, remainder);
    }
}

template <typename PrecVector>
void
Precond<PrecVector>::partBandedSweepsH(PrecVector& v)
{
    PrecVectorH sol_h = v;

    int numPartitions = m_numPartitions;
    int partSize  = m_n / numPartitions;
    int remainder = m_n % numPartitions;

    omp_set_num_threads(std::min(numPartitions, 8));

#pragma omp parallel for if (numPartitions > 4) shared(numPartitions, partSize, remainder, sol_h)
    for (int p = 0; p < numPartitions; p++) {
        int start_idx = 0, end_idx = 0;

        if (p < remainder) {
            start_idx = p * (partSize + 1);
            end_idx = start_idx + (partSize + 1);
        } else {
            start_idx = p * partSize + remainder;
            end_idx = start_idx + partSize;
        }

        int cur_half_bandwidth = m_ks_host[p];
        int pivotIdx = m_BOffsets_host[p] + cur_half_bandwidth;
        int delta = (cur_half_bandwidth << 1) + 1;

        if (m_isSPD) {
            pivotIdx -= cur_half_bandwidth;
            delta -= cur_half_bandwidth;
        }

        for (int i = start_idx; i < end_idx; i++, pivotIdx += delta) {
            PrecValueType tmp_val = sol_h[i];
            // int last_row = i + m_ks_col_host[i] + 1;
            int last_row = i + cur_half_bandwidth + 1;
            if (last_row > end_idx)
                last_row = end_idx;

            for (int j = i + 1; j < last_row; j++)
                sol_h[j] -= tmp_val * m_Bh[pivotIdx + j - i];
        }

        pivotIdx = m_BOffsets_host[p] + (m_isSPD ? 0 : cur_half_bandwidth);

        for (int i = start_idx; i < end_idx; i++, pivotIdx += delta)
            sol_h[i] /= m_Bh[pivotIdx];

        pivotIdx -= delta;
        if (m_isSPD) {
            pivotIdx -= delta;
            for (int i = end_idx - 2; i >= start_idx; i--, pivotIdx -= delta) {
                PrecValueType tmp_val = sol_h[i];

                int last_row = i + m_ks_col_host[i] + 1;
                if (last_row > m_n)
                    last_row = m_n;

                for (int j = i + 1; j < last_row; j++)
                    tmp_val -= sol_h[j] * m_Bh[pivotIdx + j - i];

                sol_h[i] = tmp_val;
            }
        } else  {
            for (int i = end_idx - 1; i > start_idx; i--, pivotIdx -= delta) {
                PrecValueType tmp_val = sol_h[i];

                int last_row = i - cur_half_bandwidth - 1;
                if (last_row < start_idx)
                    last_row = start_idx - 1;

                for (int j = i - 1; j > last_row; j--)
                    sol_h[j] -= tmp_val * m_Bh[pivotIdx + j - i];
            }
        } // end else
    } // end for
    v = sol_h;
}

/**
 * This function performs forward elimination and backward substitution
 * sweep for the given sparse matrix Acsr and vector v.
 */
template <typename PrecVector>
void 
Precond<PrecVector>::sparseSweep(PrecVector&  v,
                                 PrecVector&  w)
{
    PrecVectorH sol_h = v;

    int numPartitions = m_numPartitions;
    int partSize  = m_n / numPartitions;
    int remainder = m_n % numPartitions;
    omp_set_num_threads(std::min(16, numPartitions));

    bool last_partition_reverse = (m_numPartitions > 1 && !m_variableBandwidth);

#pragma omp parallel for shared (numPartitions, remainder, partSize, sol_h)
    for (int p = 0; p < numPartitions; p++) {
        int start_row = 0, end_row = 0;
        if (p < remainder) {
            start_row = p * (partSize + 1);
            end_row = start_row + (partSize + 1);
        } else {
            start_row = p * (partSize) + remainder;
            end_row = start_row + (partSize);
        }

        if (p == numPartitions - 1 && last_partition_reverse) {
            for (int i = end_row - 2; i >= start_row; i--) {
                int start_idx = m_Acsrh.row_offsets[i], end_idx = m_Acsrh.row_offsets[i+1];
                PrecValueType tmp_val = sol_h[i];
                for (int l = end_idx - 1; l >= start_idx; l--) {
                    int cur_k = m_Acsrh.column_indices[l];
                    if (cur_k <= i)
                        break;
                    tmp_val -= sol_h[cur_k] * m_Acsrh.values[l];
                }
                sol_h[i] = tmp_val;
            }
        } else {
            for (int i=start_row + 1; i<end_row; i++) {
                int start_idx = m_Acsrh.row_offsets[i], end_idx = m_Acsrh.row_offsets[i+1];
                PrecValueType tmp_val = sol_h[i];
                for (int l = start_idx; l < end_idx; l++) {
                    int cur_k = m_Acsrh.column_indices[l];
                    if (cur_k >= i)
                        break;
                    tmp_val -= sol_h[cur_k] * m_Acsrh.values[l];
                }
                sol_h[i] = tmp_val;
            }
        }
    }

    thrust::transform(sol_h.begin(), sol_h.end(), m_pivots.begin(), sol_h.begin(), thrust::divides<PrecValueType>());

#pragma omp parallel for shared (numPartitions, remainder, partSize, sol_h)
    for (int p = numPartitions - 1; p >= 0; p--) {
        int start_row = 0, end_row = 0;
        if (p < remainder) {
            start_row = p * (partSize + 1);
            end_row = start_row + (partSize + 1);
        } else {
            start_row = p * (partSize) + remainder;
            end_row = start_row + (partSize);
        }

        if (p == numPartitions - 1 && last_partition_reverse) {
            for (int i = start_row + 1; i < end_row; i++) {
                int start_idx = m_Acsrh.row_offsets[i], end_idx = m_Acsrh.row_offsets[i+1];
                PrecValueType tmp_val = sol_h[i];

                for (int l = start_idx; l < end_idx; l++) {
                    int cur_k = m_Acsrh.column_indices[l];
                    if (cur_k >= i)
                        break;
                    tmp_val -= sol_h[cur_k] * m_Acsrh.values[l];
                }
                sol_h[i] = tmp_val;
            }
        } else {
            for (int i = end_row - 2; i >= start_row; i--) {
                int start_idx = m_Acsrh.row_offsets[i], end_idx = m_Acsrh.row_offsets[i+1];
                PrecValueType tmp_val = sol_h[i];

                for (int l = end_idx - 1; l >= start_idx; l--) {
                    int cur_k = m_Acsrh.column_indices[l];
                    if (cur_k <= i)
                        break;
                    tmp_val -= sol_h[cur_k] * m_Acsrh.values[l];
                }
                sol_h[i] = tmp_val;
            }
        }
    }

    w = sol_h;
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

#if 0
    if (!m_variableBandwidth) {
        if (m_k > 512)
            device::forwardElimLNormal_g512<PrecValueType><<<grids, 512>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
        else
            device::forwardElimLNormal<PrecValueType><<<grids, 2*m_k-1>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
    } else
#endif
    {
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

#if 0
    if (!m_variableBandwidth && m_ilu_level < 0) {
        if (m_k > 512)
            device::backwardElimUNormal_g512<PrecValueType><<<grids, 512>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
        else
            device::backwardElimUNormal<PrecValueType><<<grids, 2*m_k-1>>>(m_n, m_k, 2*m_k, p_R, p_v, partSize, remainder);
    } else
#endif
    {
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

    if (m_ilu_level >= 0) {
        calculateSpikes_var(WV);
        return;
    }

    int totalRHSCount = cusp::blas::nrm1(m_offDiagWidths_right_host) + cusp::blas::nrm1(m_offDiagWidths_left_host);
    //// if (totalRHSCount >= 2800) {
    calculateSpikes_var(WV);
        //// return;
    ////}

    ////calculateSpikes_var_old(WV);
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

        PrecVector extV(m_k * n_eff, (PrecValueType)0), buffer;

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

                    int column_width = m_ks_host[i] + 1;
                    int delta = 0;

                    if (!m_saveMem) {
                        column_width += m_ks_host[i];
                        delta = m_ks_host[i];
                    }

                    int tmp_first_row = m_first_rows_host[i];
                    device::var::fwdElim_rightSpike_per_partition<PrecValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>> (n_eff, m_ks_host[i], m_BOffsets_host[i]+ delta +column_width*(m_first_rows_host[i]-pseudo_first_row), p_B, p_extV, m_first_rows_host[i], last_row, m_saveMem);
                    
                    int blockX = last_row - m_first_rows_host[i];
                    int gridX = 1;
                    kernelConfigAdjust(blockX, gridX, BLOCK_SIZE);
                    dim3 grids(gridX, m_offDiagWidths_right_host[i]);
                    device::var::preBck_rightSpike_divide_per_partition<PrecValueType><<<grids, blockX>>> (n_eff, m_ks_host[i], m_BOffsets_host[i]+delta+column_width*(m_first_rows_host[i]-pseudo_first_row), p_B, p_extV, m_first_rows_host[i], last_row, m_saveMem);

                    m_first_rows_host[i] = thrust::reduce(m_secondPerm.begin()+(last_row-m_k), m_secondPerm.begin()+last_row, last_row, thrust::minimum<int>());
                    device::var::bckElim_rightSpike_per_partition<PrecValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>> (n_eff, m_ks_host[i], m_BOffsets_host[i]+delta+column_width*(last_row-pseudo_first_row-1), p_B, p_extV, m_first_rows_host[i], last_row, m_saveMem);

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

        PrecVector extW(m_k * n_eff, (PrecValueType)0), buffer;

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
                    int delta = 0;
                    int column_width = m_ks_host[i+1] + 1;
                    if (!m_saveMem) {
                        delta = m_ks_host[i+1];
                        column_width += m_ks_host[i+1];
                    }

                    device::var::fwdElim_leftSpike_per_partition<PrecValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>> (n_eff, m_ks_host[i+1], m_k - m_offDiagWidths_left_host[i], m_BOffsets_host[i+1]+delta, p_B, p_extW, first_row, last_row, m_saveMem);
                    
                    int blockX = last_row - first_row;
                    int gridX = 1;
                    kernelConfigAdjust(blockX, gridX, BLOCK_SIZE);
                    dim3 grids(gridX, m_offDiagWidths_left_host[i]);

                    device::var::preBck_leftSpike_divide_per_partition<PrecValueType><<<grids, blockX>>> (n_eff, m_ks_host[i+1], m_k - m_offDiagWidths_left_host[i], m_BOffsets_host[i+1]+delta, p_B, p_extW, first_row, last_row, m_saveMem);
                    device::var::bckElim_leftSpike_per_partition<PrecValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>>(n_eff, m_ks_host[i+1], m_k - m_offDiagWidths_left_host[i], m_BOffsets_host[i+1] + delta + column_width*(last_row-first_row-1), p_B, p_extW, first_row, last_row, m_saveMem);

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

    if (m_ilu_level >= 0)
    {
        int n_eff             = m_n;
        int numPart_eff       = m_numPartitions;
        int rightOffDiagWidth = cusp::blas::nrmmax(m_offDiagWidths_right);
        int leftOffDiagWidth  = cusp::blas::nrmmax(m_offDiagWidths_left);

        PrecVector extWV((size_t)(leftOffDiagWidth + rightOffDiagWidth) * m_k * numPart_eff, (PrecValueType) 0);

        PrecValueType* p_extWV               = thrust::raw_pointer_cast(&extWV[0]);
        int*           p_ks                  = thrust::raw_pointer_cast(&m_ks[0]);
        int*           p_offDiagWidths_right = thrust::raw_pointer_cast(&m_offDiagWidths_right[0]);
        int*           p_offDiagPerms_right  = thrust::raw_pointer_cast(&m_offDiagPerms_right[0]);
        int*           p_offDiagWidths_left  = thrust::raw_pointer_cast(&m_offDiagWidths_left[0]);
        int*           p_offDiagPerms_left   = thrust::raw_pointer_cast(&m_offDiagPerms_left[0]);
        int*           p_offsets             = thrust::raw_pointer_cast(&m_BOffsets[0]);

        dim3 gridsCopy((leftOffDiagWidth + rightOffDiagWidth), numPart_eff);

        int numThreadsToUse = adjustNumThreads(cusp::blas::nrm1(m_ks) / numPart_eff);
        device::copyWVFromOrToExtendedWVTranspose_general<PrecValueType><<<gridsCopy, numThreadsToUse>>>(leftOffDiagWidth + rightOffDiagWidth, m_k, rightOffDiagWidth, m_k, 0, m_k-rightOffDiagWidth-leftOffDiagWidth, p_WV, p_extWV, false);

        {
            const int SWEEP_MAX_NUM_THREADS = 128;
            int sweepBlockX = leftOffDiagWidth;
            int sweepGridX = 1;
            if (sweepBlockX < rightOffDiagWidth)
                sweepBlockX = rightOffDiagWidth;
            kernelConfigAdjust(sweepBlockX, sweepGridX, SWEEP_MAX_NUM_THREADS);
            dim3 sweepGrids(sweepGridX, 2*numPart_eff-2);

            PrecMatrixCsr Acsr_lu      = m_Acsrh;
            PrecMatrixCsr Acsr_ul      = m_Acsrh_ul;
            PrecVector    pivots       = m_pivots;
            PrecVector    pivots_ul    = m_pivots_ul;
            int *lu_row_offsets        = thrust::raw_pointer_cast(&(Acsr_lu.row_offsets[0]));
            int *ul_row_offsets        = thrust::raw_pointer_cast(&(Acsr_ul.row_offsets[0]));
            int *lu_column_indices     = thrust::raw_pointer_cast(&(Acsr_lu.column_indices[0]));
            int *ul_column_indices     = thrust::raw_pointer_cast(&(Acsr_ul.column_indices[0]));
            PrecValueType *lu_values   = thrust::raw_pointer_cast(&(Acsr_lu.values[0]));
            PrecValueType *ul_values   = thrust::raw_pointer_cast(&(Acsr_ul.values[0]));
            PrecValueType *p_pivots    = thrust::raw_pointer_cast(&pivots[0]);
            PrecValueType *p_pivots_ul = thrust::raw_pointer_cast(&pivots_ul[0]);

            device::sparse::fwdElim_spike_partial<PrecValueType><<<sweepGrids, sweepBlockX>>>(n_eff, m_k, numPart_eff, leftOffDiagWidth + rightOffDiagWidth, lu_row_offsets, lu_column_indices, lu_values, ul_row_offsets, ul_column_indices, ul_values, p_extWV, p_offDiagWidths_left, p_offDiagWidths_right);

            int preBckBlockX = leftOffDiagWidth + rightOffDiagWidth;
            int preBckGridX  = 1;
            int preBckGridY  = m_k * numPart_eff;
            int preBckGridZ  = 1;
            kernelConfigAdjust(preBckBlockX, preBckGridX, BLOCK_SIZE);
            kernelConfigAdjust(preBckGridY, preBckGridZ, MAX_GRID_DIMENSION);
            dim3 preBckGrids(preBckGridX, preBckGridY, preBckGridZ);

            device::sparse::preBck_offDiag_divide_partial<PrecValueType><<<preBckGrids, preBckBlockX>>>(n_eff, m_k, numPart_eff, leftOffDiagWidth + rightOffDiagWidth, rightOffDiagWidth, p_pivots, p_pivots_ul, p_extWV);

            device::sparse::bckElim_spike_partial<PrecValueType><<<sweepGrids, sweepBlockX>>>(n_eff, m_k, numPart_eff, leftOffDiagWidth + rightOffDiagWidth, lu_row_offsets, lu_column_indices, lu_values, ul_row_offsets, ul_column_indices, ul_values, p_extWV, p_offDiagWidths_left, p_offDiagWidths_right);
        }

        device::copyWVFromOrToExtendedWVTranspose_general<PrecValueType><<<gridsCopy, numThreadsToUse>>>(leftOffDiagWidth + rightOffDiagWidth, m_k, rightOffDiagWidth, m_k, 0, m_k-rightOffDiagWidth-leftOffDiagWidth, p_WV, p_extWV, true);
        PrecVector WV_spare(m_k*m_k);

        PrecValueType* p_WV_spare = thrust::raw_pointer_cast(&WV_spare[0]);

        for (int i = 0; i < numPart_eff - 1; i++) {
            cusp::blas::fill(WV_spare, (PrecValueType) 0);
            device::matrixVReordering_perPartition<PrecValueType><<<m_offDiagWidths_right_host[i], numThreadsToUse>>>(m_k, p_WV+2*i*m_k*m_k, p_WV_spare, p_offDiagPerms_right+i*m_k);
            thrust::copy(WV_spare.begin(), WV_spare.end(), WV.begin()+(2*i*m_k*m_k));

            cusp::blas::fill(WV_spare, (PrecValueType) 0);
            device::matrixWReordering_perPartition<PrecValueType><<<m_offDiagWidths_left_host[i], numThreadsToUse>>>(m_k, p_WV+(2*i+1)*m_k*m_k, p_WV_spare, p_offDiagPerms_left+i*m_k);
            thrust::copy(WV_spare.begin(), WV_spare.end(), WV.begin()+((2*i+1)*m_k*m_k));
        }

        return;
    }


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

        PrecVector extWV((size_t)(leftOffDiagWidth + rightOffDiagWidth) * n_eff, (PrecValueType) 0);
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

        buffer.resize((size_t)(leftOffDiagWidth + rightOffDiagWidth) * n_eff);
        
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

            PrecMatrixCsr Acsr;
            int *row_offsets;   
            int *column_indices;
            PrecValueType *values;

            if (m_ilu_level >= 0) {
                Acsr           = m_Acsrh;
                row_offsets    = thrust::raw_pointer_cast(&(Acsr.row_offsets[0]));
                column_indices = thrust::raw_pointer_cast(&(Acsr.column_indices[0]));
                values         = thrust::raw_pointer_cast(&(Acsr.values[0]));
            }

            if (m_ilu_level < 0)
                device::var::fwdElim_spike<PrecValueType><<<sweepGrids, sweepBlockX>>>(n_eff, p_ks, leftOffDiagWidth + rightOffDiagWidth, rightOffDiagWidth, p_offsets, p_B, p_buffer, partSize, remainder, p_offDiagWidths_left, p_offDiagWidths_right, p_first_rows, m_saveMem);
            else
                device::sparse::fwdElim_spike<PrecValueType><<<sweepGrids, sweepBlockX>>>(n_eff, m_numPartitions, leftOffDiagWidth + rightOffDiagWidth, row_offsets, column_indices, values, p_buffer, p_offDiagWidths_left, p_offDiagWidths_right, p_first_rows);

            int preBckBlockX = leftOffDiagWidth + rightOffDiagWidth;
            int preBckGridX  = 1;
            int preBckGridY  = n_eff;
            int preBckGridZ  = 1;
            kernelConfigAdjust(preBckBlockX, preBckGridX, BLOCK_SIZE);
            kernelConfigAdjust(preBckGridY, preBckGridZ, MAX_GRID_DIMENSION);
            dim3 preBckGrids(preBckGridX, preBckGridY, preBckGridZ);

            if (m_ilu_level < 0)
                device::var::preBck_offDiag_divide<PrecValueType><<<preBckGrids, preBckBlockX>>>(n_eff, leftOffDiagWidth + rightOffDiagWidth, p_ks, p_offsets, p_B, p_buffer, partSize, remainder, m_saveMem);
            else {
                PrecVector pivots = m_pivots;
                PrecValueType *d_pivots = thrust::raw_pointer_cast(&pivots[0]);
                device::sparse::preBck_offDiag_divide<PrecValueType><<<preBckGrids, preBckBlockX>>>(n_eff, leftOffDiagWidth + rightOffDiagWidth, d_pivots, p_buffer);
            }

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

            if (m_ilu_level < 0)
                device::var::bckElim_spike<PrecValueType><<<sweepGrids, sweepBlockX>>>(n_eff, p_ks, leftOffDiagWidth + rightOffDiagWidth, rightOffDiagWidth, p_offsets, p_B, p_buffer, partSize, remainder, p_offDiagWidths_left, p_offDiagWidths_right, p_first_rows, m_saveMem);
            else
                device::sparse::bckElim_spike<PrecValueType><<<sweepGrids, sweepBlockX>>>(n_eff, m_numPartitions, leftOffDiagWidth + rightOffDiagWidth, row_offsets, column_indices, values, p_buffer, p_offDiagWidths_left, p_offDiagWidths_right, p_first_rows);
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

    if (m_variableBandwidth) {
        int* p_WVOffsets = thrust::raw_pointer_cast(&m_WVOffsets[0]);
        int* p_ROffsets  = thrust::raw_pointer_cast(&m_ROffsets[0]);
        int* p_spike_ks  = thrust::raw_pointer_cast(&m_spike_ks[0]);
    
        if (m_k > 1024)
            device::var::assembleReducedMat_general<PrecValueType><<<grids, 512>>>(p_spike_ks, p_WVOffsets, p_ROffsets, p_WV, p_R);
        else if (m_k > 32)
            device::var::assembleReducedMat_g32<PrecValueType><<<grids, m_k>>>(p_spike_ks, p_WVOffsets, p_ROffsets, p_WV, p_R);
        else
            device::var::assembleReducedMat<PrecValueType><<<m_numPartitions-1, m_k*m_k>>>(p_spike_ks, p_WVOffsets, p_ROffsets, p_WV, p_R);
    } else {
        if (m_k > 1024)
            device::assembleReducedMat_general<PrecValueType><<<grids, 512>>>(m_k, p_WV, p_R);
        else if (m_k > 32)
            device::assembleReducedMat_g32<PrecValueType><<<grids, m_k>>>(m_k, p_WV, p_R);
        else
            device::assembleReducedMat<PrecValueType><<<m_numPartitions-1, m_k*m_k>>>(m_k, p_WV, p_R);
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
template <typename PrecVector>
bool
Precond<PrecVector>::hasZeroPivots(const PrecVectorIterator&    start_B,
                                   const PrecVectorIterator&    end_B,
                                   int                          k,
                                   int                          step,
                                   PrecValueType                threshold)
{
    // Create a strided range to select the main diagonal
    strided_range<typename PrecVector::iterator> diag(start_B + k, end_B, step);

    // Check if any of the diagonal elements is within the specified threshold
    return thrust::any_of(diag.begin(), diag.end(), SmallerThan<PrecValueType>(threshold));
}

/**
 * This function put p largest elements at the beginning p positions of
 * an array. This is used in ILUT(p, tau) factorization algorithm.
 */
template <typename PrecVector>
void
Precond<PrecVector>::findPthMax(const IntHIterator&          ibegin,
                                const IntHIterator&          iend,
                                const PrecHIterator&         vbegin,
                                const PrecHIterator&         vend,
                                int                          p)
{
    int dist = iend - ibegin;
    if (p <= 0 || p >= dist)
        return;

    if (dist == 2) {
        if (fabs(*(vend-1)) > fabs(*vbegin)) {
            int tmp_int = *(iend - 1);
            *(iend - 1) = *ibegin;
            *ibegin = tmp_int;
            PrecValueType tmp_val = *(vend - 1);
            *(vend - 1) = *vbegin;
            *vbegin = tmp_val;
        }
        return;
    }

    int x1 = rand() % dist, x2 = rand() % dist, x3 = rand() % dist;
    if (fabs(*(vbegin + x1)) > fabs(*(vbegin + x2))) {
        x1 ^= x2;
        x2 ^= x1;
        x1 ^= x2;
    }
    if (fabs(*(vbegin + x2)) > fabs(*(vbegin + x3))) {
        x3 ^= x2;
        x2 ^= x3;
        x3 ^= x2;
    }

    int pivot_column = *(ibegin + x2);
    PrecValueType pivot_val = *(vbegin + x2);

    if (vbegin + x2 != vend - 1) {
        *(ibegin + x2) ^= *(iend - 1);
        *(iend - 1) ^= *(ibegin + x2);
        *(ibegin + x2) ^= *(iend - 1);
        PrecValueType tmp_val = *(vend - 1);
        *(vend - 1) = *(vbegin + x2);
        *(vbegin + x2)= tmp_val;
    }

    x1 = 0;
    x3 = x2 = (vend - vbegin) - 1;

    while(true) {
        while (x1 < x3 && fabs(*(vbegin + x1)) >= fabs(pivot_val)) x1++;

        if (x1 == x3) {
            *(vbegin + x2) = pivot_val;
            *(ibegin + x2) = pivot_column;
            break;
        }
        *(vbegin + x2) = *(vbegin + x1);
        *(ibegin + x2) = *(ibegin + x1);
        x2 = x1;
        x3 --;

        while (x1 < x3 && fabs(*(vbegin + x3)) < fabs(pivot_val)) x3--;
        if (x1 == x3) {
            *(vbegin + x2) = pivot_val;
            *(ibegin + x2) = pivot_column;
            break;
        }
        *(vbegin + x2) = *(vbegin + x3);
        *(ibegin + x2) = *(ibegin + x3);
        x2 = x3;
        x1++;
    }

    if (x2 == p || x2 == p - 1)
        return;

    if (x2 > p)
        findPthMax(ibegin, ibegin + x2, vbegin, vbegin + x2, p);
    else
        findPthMax(ibegin + (x2 + 1), iend, vbegin + (x2 + 1), vend, p - x2 - 1);
}



} // namespace sap


#endif
