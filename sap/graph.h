#ifndef SAP_GRAPH_H
#define SAP_GRAPH_H

#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <algorithm>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif
#include <cusp/print.h>

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/gather.h>

#include <sap/common.h>
#include <sap/timer.h>
#include <sap/device/data_transfer.cuh>
#include <sap/device/db.cuh>

#include <sap/exception.h>

namespace sap {

template <typename T>
class Graph
{
public:
	typedef typename cusp::coo_matrix<int, T, cusp::host_memory> MatrixCoo;
	typedef typename cusp::csr_matrix<int, T, cusp::host_memory> MatrixCsr;
	typedef typename cusp::csr_matrix<int, T, cusp::device_memory>  MatrixCsrD;
	typedef typename cusp::array1d<T, cusp::host_memory>         Vector;
	typedef typename cusp::array1d<T, cusp::device_memory>       VectorD;
	typedef typename cusp::array1d<double, cusp::host_memory>    DoubleVector;
	typedef typename cusp::array1d<int, cusp::host_memory>       IntVector;
	typedef typename cusp::array1d<int, cusp::device_memory>     IntVectorD;
	typedef typename cusp::array1d<double, cusp::device_memory>  DoubleVectorD;
	typedef typename cusp::array1d<bool, cusp::host_memory>      BoolVector;
	typedef typename cusp::array1d<bool, cusp::device_memory>    BoolVectorD;
	typedef Vector                                               MatrixMapF;
	typedef IntVector                                            MatrixMap;

	typedef typename IntVector::iterator                         IntIterator;
	typedef typename Vector::iterator                            TIterator;
	typedef typename IntVector::reverse_iterator                 IntRevIterator;
	typedef typename Vector::reverse_iterator                    TRevIterator;
	typedef typename thrust::tuple<IntIterator, IntIterator>     IntIteratorTuple;
	typedef typename thrust::tuple<IntIterator, IntIterator, TIterator>     IteratorTuple;
	typedef typename thrust::tuple<IntRevIterator, IntRevIterator, TRevIterator>   RevIteratorTuple;
	typedef typename thrust::zip_iterator<IntIteratorTuple>      EdgeIterator;
	typedef typename thrust::zip_iterator<IteratorTuple>         WeightedEdgeIterator;
	typedef typename thrust::zip_iterator<RevIteratorTuple>      WeightedEdgeRevIterator;

	typedef typename thrust::tuple<int, int>                     NodeType;
	typedef typename thrust::tuple<int, int, T>                  WeightedEdgeType;
	typedef typename thrust::tuple<int, double>                  Dijkstra;

private:
	typedef enum {
		INACTIVE,
		PREACTIVE,
		ACTIVE,
		POSTACTIVE
	} Status;
	typedef typename cusp::array1d<Status, cusp::host_memory>    StatusVector;

public:
	Graph(bool trackReordering = false);

	double     getTimeDB() const     {return m_timeDB;}
	double     getTimeDBPre() const     {return m_timeDB_pre;}
	double     getTimeDBFirst() const     {return m_timeDB_first;}
	double     getTimeDBSecond() const     {return m_timeDB_second;}
	double     getTimeDBPost() const     {return m_timeDB_post;}
	double     getTimeRCM() const      {return m_timeRCM;}
	double     getTimeDropoff() const  {return m_timeDropoff;}

	int        reorder(const MatrixCsr& Acsr,
	                   bool             testDB,
	                   bool             doDB,
					   bool             dbFirstStageOnly,
	                   bool             scale,
					   bool             doRCM,
					   bool             doSloan,
	                   IntVector&       optReordering,
	                   IntVector&       optPerm,
	                   IntVectorD&      d_dbRowPerm,
	                   VectorD&         d_dbRowScale,
	                   VectorD&         d_dbColScale,
	                   MatrixMapF&      scaleMap,
	                   int&             k_db);

	int        dropOff(T   frac,
	                   int maxBandwidth,
	                   T&  frac_actual);

	void       assembleOffDiagMatrices(int         bandwidth,
	                                   int         numPartitions,
	                                   Vector&     WV_host,
	                                   Vector&     offDiags_host,
	                                   IntVector&  offDiagWidths_left,
	                                   IntVector&  offDiagWidths_right,
	                                   IntVector&  offDiagPerms_left,
	                                   IntVector&  offDiagPerms_right,
	                                   MatrixMap&  typeMap,
	                                   MatrixMap&  offDiagMap,
	                                   MatrixMap&  WVMap);

	void       secondLevelReordering(int         bandwidth,
	                                 int         numPartitions,
	                                 IntVector&  secondReorder,
	                                 IntVector&  secondPerm,
	                                 IntVector&  first_rows);

	void       assembleBandedMatrix(int         bandwidth,
	                                IntVector&  ks_col,
	                                IntVector&  ks_row,
									MatrixCoo&  Acoo,
	                                MatrixMap&  typeMap,
	                                MatrixMap&  bandedMatMap);

	void       assembleBandedMatrix(int         bandwidth,
									bool        saveMem,
	                                int         numPartitions,
	                                IntVector&  ks_col,
	                                IntVector&  ks_row,
									MatrixCoo&  Acoo,
	                                IntVector&  ks,
	                                IntVector&  BOffsets,
	                                MatrixMap&  typeMap,
	                                MatrixMap&  bandedMatMap);

	void       get_csr_matrix(MatrixCsr&        Acsr,
							  int               numPartitions);

	void       unorderedBFS(bool          doRCM,
			        		bool          doSloan,
							IntVector&    tmp_reordering,
							IntVector&    row_offsets,
							IntVector&    column_indices,
							IntVector&    visited,
							IntVector&    levels,
							IntVector&    ori_degrees,
							BoolVector&   tried);

	void       unorderedBFSIteration(int            width,
							         int            start_idx,
									 int            end_idx,
									 IntVector&     tmp_reordering,
									 IntVector&     levels,
									 IntVector&     visited,
									 IntVector&     row_offsets,
									 IntVector&     column_indices,
									 IntVector&     ori_degrees,
									 BoolVector&    tried,
									 IntVector&     costs,
									 IntVector&     ori_costs,
									 StatusVector&  status,
									 int &          next_level);

private:
	int           m_n;
	int           m_nnz;
	MatrixCsr     m_matrix;
	MatrixCsr     m_matrix_diagonal;
	IntVector     m_ori_indices;
	IntVector     m_ori_indices_diagonal;

	bool          m_trackReordering;

	double        m_timeDB;
	double        m_timeDB_pre;
	double        m_timeDB_first;
	double        m_timeDB_second;
	double        m_timeDB_post;
	double        m_timeRCM;
	double        m_timeDropoff;

	BoolVector    m_exists;

	// Temporarily used in partitioned RCM for buffering
	IntVector     m_buffer_reordering;

	// Temporarily used in the third stage of DB for buffering
	IntVector     m_DB_B;
	BoolVector    m_DB_inB;
	DoubleVector  m_DB_d_vals;
	BoolVector    m_DB_visited;

	bool       DB(const MatrixCsr& Acsr,
			      bool             scale,
				  bool             dbFirstStageOnly,
	              IntVectorD&      dbRowPerm,
	              DoubleVectorD&   dbRowScale,
	              DoubleVectorD&   dbColScale,
	              MatrixMapF&      scaleMap);

	int        RCM(MatrixCsr&   matcsr,
	               IntVector&   optReordering,
	               IntVector&   optPerm);

	bool       partitionedRCM(MatrixCsr&     mat_csr,
							  int            index_begin,
							  int            index_end,
	                          int            node_begin,
	                          int            node_end,
	                          IntVector&     optReordering,
	                          IntVector&     optPerm,
							  IntVector&     row_offsets,
							  IntVector&     row_indices);

	void       buildTopology(EdgeIterator&      begin,
							 EdgeIterator&      end,
							 int                node_begin,
							 int                node_end,
	                         IntVector&         row_offsets,
							 IntVector&         column_indices);

	static const double LOC_INFINITY;

	void       init_reduced_cval(bool           first_stage_only,
			                     const IntVector&     row_ptr,
	                             const IntVector&     rows,
	                             DoubleVector&  c_val, 
	                             DoubleVector&  u_val,
	                             DoubleVector&  v_val,
	                             IntVector&     match_nodes,
	                             IntVector&     rev_match_nodes,
	                             BoolVector&    matched,
	                             BoolVector&    rev_matched);
	bool       find_shortest_aug_path(int init_node,
	                                  BoolVector& matched, BoolVector& rev_matched, 
	                                  IntVector& match_nodes, IntVector& rev_match_nodes,
	                                  const IntVector& row_ptr, const IntVector& rows, IntVector& prev,
	                                  DoubleVector& u_val,
	                                  DoubleVector& v_val,
	                                  DoubleVector& c_val,
	                                  IntVector&    irn);
	void       get_csc_matrix(const MatrixCsr&  Acsr,
	                          DoubleVectorD&    c_val,
	                          DoubleVectorD&    max_val_in_col);

	int        sloan(MatrixCsr&   matcsr,
	                 IntVector&   optReordering,
	                 IntVector&   optPerm);

	size_t     symbolicFactorization(const MatrixCsr&  Acsr);

public:
	template <typename VType>
	struct AbsoluteValue: public thrust::unary_function<VType, VType>
	{
		__host__ __device__
		VType operator() (VType a)
		{
			return (a < 0 ? -a : a);
		}
	};

	struct Square: public thrust::unary_function<T, T>
	{
		__host__ __device__
		T operator() (T a)
		{
			return a*a;
		}
	};

	struct AccumulateEdgeWeights: public thrust::unary_function<WeightedEdgeType, T>
	{
		inline __host__ __device__
		T operator() (WeightedEdgeType a)
		{
			return thrust::get<2>(a) * thrust::get<2>(a);
		}
	};

	struct is_not
	{
		__host__ __device__
		bool operator() (bool x) {
			return !x;
		}
	};

	struct ClearValue: public thrust::unary_function<double, double>
	{
		__host__ __device__
		double operator() (double a)
		{
			return 0.0;
		}
	};

	struct Exponential: public thrust::unary_function<double, double>
	{
		__host__ __device__
		double operator() (double a)
		{
			return exp(a);
		}
	};

	struct EdgeLength: public thrust::unary_function<NodeType, int>
	{
		__host__ __device__
		int operator() (NodeType a)
		{
			int row = thrust::get<0>(a), col = thrust::get<1>(a);
			int diff = row - col;
			return (diff < 0 ? -diff : diff);
		}
	};

	struct PermutedEdgeLength: public thrust::unary_function<NodeType, int>
	{
		int *m_perm_array;
		PermutedEdgeLength(int *perm_array):m_perm_array(perm_array) {}
		__host__ __device__
		int operator() (NodeType a)
		{
			int row = m_perm_array[thrust::get<0>(a)], col = m_perm_array[thrust::get<1>(a)];
			int diff = row - col;
			return (diff < 0 ? -diff : diff);
		}
	};

	struct PermuteEdge: public thrust::unary_function<NodeType, NodeType>
	{
		int *m_perm_array;
		PermuteEdge(int *perm_array):m_perm_array(perm_array) {}
		__host__ __device__
		NodeType operator() (NodeType a)
		{
			return thrust::make_tuple(m_perm_array[thrust::get<0>(a)], m_perm_array[thrust::get<1>(a)]);
		}
	};

	template <typename Type>
	struct Map: public thrust::unary_function<int, Type>
	{
		Type *m_map;
		Map(Type *arg_map): m_map(arg_map) {}
		__host__ __device__
		Type operator() (int a) { return m_map[a];}
	};

	struct GetCount: public thrust::unary_function<int, int>
	{
		__host__ __device__
		int operator() (int a)
		{
			return 1;
		}
	};

	template <typename VType>
	struct CompareValue
	{
		bool operator () (const thrust::tuple<int, VType> &a, const thrust::tuple<int, VType> &b) const {return thrust::get<1>(a) > thrust::get<1>(b);}
	};

	struct Difference: public thrust::binary_function<int, int, int>
	{
		inline
		__host__ __device__
		int operator() (const int &a, const int &b) const {
			return abs(a-b);
		}
	};

	template <typename Type>
	struct EqualTo: public thrust::unary_function<Type, bool>
	{
		Type m_local;
		EqualTo(Type l): m_local(l) {}
		inline
		__host__ __device__
		bool operator() (const Type& a) {
			return a == m_local;
		}
	};
};


template <typename T>
const double Graph<T>::LOC_INFINITY = 1e37;


// ----------------------------------------------------------------------------
// Graph::Graph()
//
// This is the constructor for the Graph class.
// ----------------------------------------------------------------------------
template <typename T>
Graph<T>::Graph(bool trackReordering)
:	m_timeDB(0),
	m_timeDB_pre(0),
	m_timeDB_first(0),
	m_timeDB_second(0),
	m_timeDB_post(0),
	m_timeRCM(0),
	m_timeDropoff(0),
	m_trackReordering(trackReordering)
{
}


// ----------------------------------------------------------------------------
// Graph::reorder()
//
// This function applies various reordering algorithms to the specified matrix
// (assumed to be in COO format and on the host) for bandwidth reduction and
// diagonal boosting. It returns the half-bandwidth after reordering.
// ----------------------------------------------------------------------------
template <typename T>
int
Graph<T>::reorder(const MatrixCsr&  Acsr,
                  bool              testDB,
                  bool              doDB,
				  bool              dbFirstStageOnly,
                  bool              scale,
				  bool              doRCM,
				  bool              doSloan,
                  IntVector&        optReordering,
                  IntVector&        optPerm,
                  IntVectorD&       d_dbRowPerm,
                  VectorD&          d_dbRowScale,
                  VectorD&          d_dbColScale,
                  MatrixMapF&       scaleMap,
                  int&              k_db)
{
	m_n = Acsr.num_rows;
	m_nnz = Acsr.num_entries;

	m_buffer_reordering.resize(m_n);

	// Apply DB algorithm. Note that we must ensure we always work with
	// double precision scale vectors.
	//
	// TODO:  how can we check if the precision of Vector is already
	//        double, so that we can save extra copies.
	if (doDB) {
		GPUTimer loc_timer;
		loc_timer.Start();
		DoubleVectorD  dbRowScaleD;
		DoubleVectorD  dbColScaleD;

		m_DB_inB.resize(m_n, false);
		m_DB_B.resize(m_n, 0);
		m_DB_d_vals.resize(m_n, LOC_INFINITY);
		m_DB_visited.resize(m_n, false);

		DB(Acsr, scale, dbFirstStageOnly, d_dbRowPerm, dbRowScaleD, dbColScaleD, scaleMap);
		d_dbRowScale = dbRowScaleD;
		d_dbColScale = dbColScaleD;
		loc_timer.Stop();
		m_timeDB = loc_timer.getElapsed();
	} else {
		d_dbRowScale.resize(m_n);
		d_dbColScale.resize(m_n);
		d_dbRowPerm.resize(m_n);
		scaleMap.resize(m_nnz);

		m_matrix = Acsr;

		thrust::sequence(d_dbRowPerm.begin(), d_dbRowPerm.end());
		cusp::blas::fill(d_dbRowScale, (T) 1.0);
		cusp::blas::fill(d_dbColScale, (T) 1.0);
		cusp::blas::fill(scaleMap, (T) 1.0);
	}

	{
		IntVector row_indices(m_nnz);
#ifdef   USE_OLD_CUSP
		cusp::detail::offsets_to_indices(m_matrix.row_offsets, row_indices);
#else
		cusp::offsets_to_indices(m_matrix.row_offsets, row_indices);
#endif
		k_db = thrust::inner_product(row_indices.begin(), row_indices.end(), m_matrix.column_indices.begin(), 0, thrust::maximum<int>(), Difference());
	}

	if (testDB)
		return k_db;

	// Apply reverse Cuthill-McKee algorithm.
	int bandwidth;
	if (doRCM)
		bandwidth = RCM(m_matrix, optReordering, optPerm);
	else if (doSloan)
		bandwidth = sloan(m_matrix, optReordering, optPerm);
	else {
		bandwidth = k_db;
		optReordering.resize(m_n);
		optPerm.resize(m_n);
		thrust::sequence(optReordering.begin(), optReordering.end());
		thrust::sequence(optPerm.begin(), optPerm.end());
	}

	// Return the bandwidth obtained after reordering.
	return bandwidth;
}


// ----------------------------------------------------------------------------
// Graph::dropOff()
//
// This function identifies the elements that can be removed in the reordered
// matrix while reducing the element-wise 1-norm by no more than the specified
// fraction. 
//
// We cache an iterator in the (now ordered) vector of edges, such that the
// edges from that point to the end encode a banded matrix whose element-wise
// 1-norm is at least (1-frac) ||A||_1.
//
// Note that the final bandwidth is guranteed to be no more than the specified
// maxBandwidth value.
//
// The return value is the actual half-bandwidth we got after drop-off.
// ----------------------------------------------------------------------------
template <typename T>
int
Graph<T>::dropOff(T   frac,
                  int maxBandwidth,
                  T&  frac_actual)
{
	CPUTimer timer;
	timer.Start();

	// Sort the edges in *decreasing* order of their length (the difference
	// between the indices of their adjacent nodes).
	// std::sort(m_edges.begin(), m_edges.end(), CompareEdgeLength());
	MatrixCoo  Acoo(m_n, m_n, m_nnz);
	IntVector  bucket(m_n, 0);

	for (int i = 0; i < m_n; i++) {
		int start_idx = m_matrix.row_offsets[i];
		int end_idx = m_matrix.row_offsets[i+1];

		for (int l = start_idx; l < end_idx; l++)
			bucket[abs(i - m_matrix.column_indices[l])] ++;
	}

	thrust::exclusive_scan(bucket.begin(), bucket.end(), bucket.begin());

	for (int i = 0; i < m_n; i++) {
		int start_idx = m_matrix.row_offsets[i];
		int end_idx = m_matrix.row_offsets[i+1];

		for (int l = start_idx; l < end_idx; l++) {
			int idx = (bucket[abs(i - m_matrix.column_indices[l])] ++);
			Acoo.row_indices[idx]    = i;
			Acoo.column_indices[idx] = m_matrix.column_indices[l];
			Acoo.values[idx]         = m_matrix.values[l];
		}
	}

	// Calculate the 1-norm of the current matrix and the minimum norm that
	// must be retained after drop-off. Initialize the 1-norm of the resulting
	// truncated matrix.
	T norm_in = thrust::transform_reduce(m_matrix.values.begin(), m_matrix.values.end(), Square(), (T)0, thrust::plus<T>());
	T min_norm_out = (1 - frac) * norm_in;
	T norm_out = norm_in;

	// Walk all edges and accumulate the weigth (1-norm) of one band at a time.
	// Continue until we are left with the main diagonal only or until the weight
	// of all proccessed bands exceeds the allowable drop off (provided we do not
	// exceed the specified maximum bandwidth.
	WeightedEdgeRevIterator first(thrust::make_tuple(Acoo.row_indices.rbegin(), Acoo.column_indices.rbegin(), Acoo.values.rbegin()));
	WeightedEdgeRevIterator last = first;
	int   final_half_bandwidth = abs(thrust::get<0>(*first) - thrust::get<1>(*first));

	// Remove all elements which are outside the specified max bandwidth
	{
		int bandwidth;
		while ((bandwidth = abs(thrust::get<0>(*last) - thrust::get<1>(*last))) > maxBandwidth)
			last++;

		norm_out -= thrust::transform_reduce(first, last, AccumulateEdgeWeights(), (T)0, thrust::plus<T>());
		first = last;
		final_half_bandwidth = bandwidth;
	}

	// After the first stage, we haven't reached the budget, drop off more.
	if (norm_out >= min_norm_out){
		while (true) {
			// Current band
			int bandwidth = abs(thrust::get<0>(*first) - thrust::get<1>(*first));

			// Stop now if we reached the main diagonal.
			if (bandwidth == 0) {
				final_half_bandwidth = bandwidth;
				break;
			}

			// Find all edges in the current band and calculate the norm of the band.
			do {last++;}  while (abs(thrust::get<0>(*last) - thrust::get<1>(*last)) == bandwidth);

			T band_norm = thrust::transform_reduce(first, last, AccumulateEdgeWeights(), T(0), thrust::plus<T>());

			// Stop now if removing this band would reduce the norm by more than allowed.
			if (norm_out - band_norm < min_norm_out)
				break;

			// Remove the norm of this band and move to the next one.
			norm_out -= band_norm;
			first = last;
			final_half_bandwidth = bandwidth;
		}
	}

	timer.Stop();
	m_timeDropoff = timer.getElapsed();

	// Calculate the actual norm reduction fraction.
	frac_actual = 1 - norm_out/norm_in;

	// Restore the matrix after drop-off
	{
		m_nnz -= first - thrust::make_zip_iterator(thrust::make_tuple(Acoo.row_indices.rbegin(), Acoo.column_indices.rbegin(), Acoo.values.rbegin()));
		Acoo.values.resize(m_nnz);
		Acoo.column_indices.resize(m_nnz);
		Acoo.row_indices.resize(m_nnz);
		Acoo.num_entries = m_nnz;

		//// m_matrix = Acoo;
		//// Acoo.sort_by_row_and_column();

		{
			thrust::fill(bucket.begin(), bucket.end(), T(0));
			for (int i = 0; i < m_nnz; i++)
				bucket[Acoo.column_indices[i]] ++;

			thrust::exclusive_scan(bucket.begin(), bucket.end(), bucket.begin());

			IntVector  tmp_row_indices(m_nnz);

			for (int i = 0; i < m_nnz; i++) {
				int idx = (bucket[Acoo.column_indices[i]]++);
				tmp_row_indices[idx] = Acoo.row_indices[i];
				m_matrix.column_indices[idx] = Acoo.column_indices[i];
				m_matrix.values[idx] = Acoo.values[i];
			}

			thrust::fill(bucket.begin(), bucket.end(), T(0));
			for (int i = 0; i < m_nnz; i++)
				bucket[tmp_row_indices[i]] ++;

			thrust::inclusive_scan(bucket.begin(), bucket.end(), bucket.begin());

			for (int i = 0; i < m_nnz; i++) {
				int idx = (--bucket[tmp_row_indices[i]]);
				Acoo.row_indices[idx]    = tmp_row_indices[i];
				Acoo.column_indices[idx] = m_matrix.column_indices[i];
				Acoo.values[idx]         = m_matrix.values[i];
			}
		}

#ifdef USE_OLD_CUSP
		cusp::detail::indices_to_offsets(Acoo.row_indices, m_matrix.row_offsets);
#else
		cusp::indices_to_offsets(Acoo.row_indices, m_matrix.row_offsets);
#endif
		m_matrix.column_indices = Acoo.column_indices;
		m_matrix.values         = Acoo.values;
		m_matrix.num_entries    = m_nnz;
	}

	return final_half_bandwidth;
}


// ----------------------------------------------------------------------------
// Graph::assembleOffDiagMatrices()
//
// This function finds all non-zeroes in off-diagonal matrices and assemble
// off-diagonal matrices.
// ----------------------------------------------------------------------------
template <typename T>
void
Graph<T>::assembleOffDiagMatrices(int         bandwidth,
                                  int         numPartitions,
                                  Vector&     WV_host,
                                  Vector&     offDiags_host,
                                  IntVector&  offDiagWidths_left,
                                  IntVector&  offDiagWidths_right,
                                  IntVector&  offDiagPerms_left,
                                  IntVector&  offDiagPerms_right,
                                  MatrixMap&  typeMap,
                                  MatrixMap&  offDiagMap,
                                  MatrixMap&  WVMap)
{
	if (WV_host.size() != 2*bandwidth*bandwidth*(numPartitions-1)) {
		WV_host.resize(2*bandwidth*bandwidth*(numPartitions-1), 0);
		offDiags_host.resize(2*bandwidth*bandwidth*(numPartitions-1), 0);
		offDiagWidths_left.resize(numPartitions-1, 0);
		offDiagWidths_right.resize(numPartitions-1, 0);

	} else {
		cusp::blas::fill(WV_host, (T) 0);
		cusp::blas::fill(offDiags_host, (T) 0);
		cusp::blas::fill(offDiagWidths_left, 0);
		cusp::blas::fill(offDiagWidths_right, 0);
	}

	offDiagPerms_left.resize((numPartitions-1) * bandwidth, -1);
	offDiagPerms_right.resize((numPartitions-1) * bandwidth, -1);

	IntVector offDiagReorderings_left((numPartitions-1) * bandwidth, -1);
	IntVector offDiagReorderings_right((numPartitions-1) * bandwidth, -1);

	int partSize = m_n / numPartitions;
	int remainder = m_n % numPartitions;

	MatrixCoo Acoo(m_n, m_n, m_nnz);
	int num_entries = 0;

	if (m_trackReordering) {
		typeMap.resize(m_nnz);
		offDiagMap.resize(m_nnz);
		WVMap.resize(m_nnz);
		m_ori_indices_diagonal.resize(m_nnz);
	}

	m_exists.resize(m_n);
	cusp::blas::fill(m_exists, false);

	for (int it2 = 0; it2 < m_n; it2++) {
		int start_idx = m_matrix.row_offsets[it2];
		int end_idx = m_matrix.row_offsets[it2+1];
		for (int it = start_idx; it < end_idx; it++)
		{
			int j = it2;
			int l = m_matrix.column_indices[it];
			int curPartNum = l / (partSize + 1);
			if (curPartNum >= remainder)
				curPartNum = remainder + (l-remainder * (partSize + 1)) / partSize;

			int curPartNum2 = j / (partSize + 1);
			if (curPartNum2 >= remainder)
				curPartNum2 = remainder + (j-remainder * (partSize + 1)) / partSize;

			if (curPartNum == curPartNum2) {
				Acoo.row_indices[num_entries]    = j;
				Acoo.column_indices[num_entries] = l;
				Acoo.values[num_entries]         = m_matrix.values[it];

				if (m_trackReordering)
					m_ori_indices_diagonal[num_entries] = m_ori_indices[it];

				num_entries++;
			}
			else {
				if (curPartNum > curPartNum2) { // V/B Matrix
					m_exists[j] = true;
					int partEndRow = partSize * curPartNum2;
					if (curPartNum2 < remainder)
						partEndRow += curPartNum2 + partSize + 1;
					else
						partEndRow += remainder + partSize;

					int partStartCol = partSize * curPartNum;
					if (curPartNum < remainder)
						partStartCol += curPartNum;
					else
						partStartCol += remainder;

					if (offDiagReorderings_right[curPartNum2*bandwidth+l-partStartCol] < 0) {
						offDiagReorderings_right[curPartNum2*bandwidth+l-partStartCol] = offDiagWidths_right[curPartNum2];
						offDiagPerms_right[curPartNum2*bandwidth+offDiagWidths_right[curPartNum2]] = l-partStartCol;
						offDiagWidths_right[curPartNum2]++;
					}

					if (m_trackReordering) {
						int ori_idx = m_ori_indices[it];
						typeMap[ori_idx] = 0;
						offDiagMap[ori_idx] = curPartNum2*2*bandwidth*bandwidth + (j+bandwidth-partEndRow) * bandwidth + (l-partStartCol);
						WVMap[ori_idx] = curPartNum2*2*bandwidth*bandwidth + (j+bandwidth-partEndRow) + offDiagReorderings_right[curPartNum2*bandwidth+l-partStartCol] * bandwidth;
					}

					offDiags_host[curPartNum2*2*bandwidth*bandwidth + (j+bandwidth-partEndRow) * bandwidth + (l-partStartCol)] = WV_host[curPartNum2*2*bandwidth*bandwidth + (j+bandwidth-partEndRow) + offDiagReorderings_right[curPartNum2*bandwidth+l-partStartCol] * bandwidth] = m_matrix.values[it];

				} else {                          // W/C Matrix
					int partStartRow = partSize * curPartNum2;
					if (curPartNum2 < remainder)
						partStartRow += curPartNum2;
					else
						partStartRow += remainder;

					int partEndCol = partSize * curPartNum;
					if (curPartNum < remainder)
						partEndCol += curPartNum + partSize + 1;
					else
						partEndCol += remainder + partSize;

					if (offDiagReorderings_left[(curPartNum2-1)*bandwidth+l-partEndCol+bandwidth] < 0) {
						offDiagReorderings_left[(curPartNum2-1)*bandwidth+l-partEndCol+bandwidth] = bandwidth - 1 - offDiagWidths_left[curPartNum2-1];
						offDiagPerms_left[(curPartNum2-1)*bandwidth+bandwidth-1-offDiagWidths_left[curPartNum2-1]] = l-partEndCol+bandwidth;
						offDiagWidths_left[curPartNum2-1]++;
					}

					if (m_trackReordering) {
						int ori_idx = m_ori_indices[it];
						typeMap[ori_idx] = 0;
						offDiagMap[ori_idx] = (curPartNum*2+1)*bandwidth*bandwidth + (j-partStartRow) * bandwidth + (l-partEndCol+bandwidth);
						WVMap[ori_idx] = (curPartNum*2+1)*bandwidth*bandwidth + (j-partStartRow) + (offDiagReorderings_left[(curPartNum2-1)*bandwidth+l-partEndCol+bandwidth]) * bandwidth;
					}

					offDiags_host[(curPartNum*2+1)*bandwidth*bandwidth + (j-partStartRow) * bandwidth + (l-partEndCol+bandwidth)] = WV_host[(curPartNum*2+1)*bandwidth*bandwidth + (j-partStartRow) + (offDiagReorderings_left[(curPartNum2-1)*bandwidth+l-partEndCol+bandwidth]) * bandwidth] = m_matrix.values[it];
				} // end else
			} // end else
		} // end for
	} // end for

	Acoo.row_indices.resize(num_entries);
	Acoo.column_indices.resize(num_entries);
	Acoo.values.resize(num_entries);
	m_ori_indices_diagonal.resize(num_entries);
	Acoo.num_entries = num_entries;
	m_matrix_diagonal = Acoo;
}


// ----------------------------------------------------------------------------
// Graph::secondLevelReordering()
//
// This function applies the second-stage reordering if the flags ``m_reorder''
// and ``m_secondLevelReordering'' are both true. Second-level reordering is
// essentially RCM on all partitions.
// ----------------------------------------------------------------------------
template <typename T>
void
Graph<T>::secondLevelReordering(int       bandwidth,
                                int       numPartitions,
                                IntVector&  secondReorder,
                                IntVector&  secondPerm,
                                IntVector&  first_rows)
{
	int node_begin = 0, node_end;
	int partSize = m_n / numPartitions;
	int remainder = m_n % numPartitions;
	int edgeBegin = 0, edgeEnd;
	secondReorder.resize(m_n);
	secondPerm.resize(m_n);

	int diagonal_nnz = m_matrix_diagonal.num_entries;

	IntVector row_indices(diagonal_nnz);
	IntVector row_offsets(m_n + 1);
#ifdef   USE_OLD_CUSP
	cusp::detail::offsets_to_indices(m_matrix_diagonal.row_offsets, row_indices);
#else
	cusp::offsets_to_indices(m_matrix_diagonal.row_offsets, row_indices);
#endif


	for (int i = 0; i < numPartitions; i++) {
		if (i < remainder)
			node_end = node_begin + partSize + 1;
		else
			node_end = node_begin + partSize;

		edgeEnd = m_matrix_diagonal.row_offsets[node_end];

		partitionedRCM(m_matrix_diagonal,
					   edgeBegin,
		               edgeEnd,
		               node_begin,
		               node_end,
		               secondReorder,
		               secondPerm,
					   row_offsets,
					   row_indices);

		node_begin = node_end;
		edgeBegin = edgeEnd;
	}

	{
		int *perm_array = thrust::raw_pointer_cast(&secondPerm[0]);
		thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), m_matrix_diagonal.column_indices.begin())),
						  thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   m_matrix_diagonal.column_indices.end())),
						  thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), m_matrix_diagonal.column_indices.begin())),
						  PermuteEdge(perm_array));

		//thrust::sort_by_key(row_indices.begin(), row_indices.end(), 
		//				    thrust::make_zip_iterator(thrust::make_tuple(m_matrix_diagonal.column_indices.begin(), m_matrix_diagonal.values.begin())));
		{
			thrust::fill(row_offsets.begin(), row_offsets.end(), 0);
			IntVector column_indices(diagonal_nnz);
			Vector    values(diagonal_nnz);
			IntVector ori_indices;

			if (m_trackReordering)
				ori_indices.resize(diagonal_nnz);

			for (int i = 0; i < diagonal_nnz; i++)
				row_offsets[row_indices[i]] ++;

			thrust::inclusive_scan(row_offsets.begin(), row_offsets.end(), row_offsets.begin());

			for (int i = diagonal_nnz - 1; i >= 0; i--) {
				int idx = (--row_offsets[row_indices[i]]);
				column_indices[idx] = m_matrix_diagonal.column_indices[i];
				values[idx] = m_matrix_diagonal.values[i];
				if (m_trackReordering)
					ori_indices[idx] = m_ori_indices_diagonal[i];
			}
			m_matrix_diagonal.column_indices = column_indices;
			m_matrix_diagonal.values         = values;
			m_matrix_diagonal.row_offsets    = row_offsets;
			if (m_trackReordering)
				m_ori_indices_diagonal = ori_indices;
		}
	}

	first_rows.resize(numPartitions - 1);

	int last_row = 0;
	for (int i=0; i<numPartitions - 1; i++) {
		if (i < remainder)
			last_row += (partSize + 1);
		else
			last_row += partSize;

		first_rows[i] = last_row;
		for (int j = last_row-bandwidth; j < last_row; j++)
			if (m_exists[j] && first_rows[i] > secondPerm[j])
				first_rows[i] = secondPerm[j];
	}
}


// ----------------------------------------------------------------------------
// Graph::assembleBandedMatrix()
//
// These functions assemble the Spike banded matrix using current information
// in the graph. The first version creates a banded matrix of constant
// bandwidth across all partitions. The second version assmebles a matrix
// that has banded diagonal blocks of different bandwidths for each partition.
// ----------------------------------------------------------------------------
template <typename T>
void
Graph<T>::assembleBandedMatrix(int         bandwidth,
                               IntVector&  ks_col,
                               IntVector&  ks_row,
							   MatrixCoo&  Acoo,
                               MatrixMap&  typeMap,
                               MatrixMap&  bandedMatMap)
{
	ks_col.resize(m_n, 0);
	ks_row.resize(m_n, 0);

	if (m_trackReordering) {
		if (typeMap.size() <= 0)
			typeMap.resize(m_nnz);
		bandedMatMap.resize(m_nnz);
	}

	int idx = 0;
	Acoo.resize(m_n, m_n, m_nnz);

	if (m_trackReordering) {
		for (int it2 = 0; it2 < m_n; it2++) {
			int start_idx = m_matrix.row_offsets[it2];
			int end_idx = m_matrix.row_offsets[it2+1];

			for (int it = start_idx; it < end_idx; ++it, ++idx) {
				int j = it2;
				int l = m_matrix.column_indices[it];

				size_t i = (size_t) l * (2 * bandwidth + 1) + bandwidth + j - l;
				int ori_idx = m_ori_indices[it];
				typeMap[ori_idx] = 1;
				bandedMatMap[ori_idx] = i;

				Acoo.row_indices[idx] = j;
				Acoo.column_indices[idx] = l;
				Acoo.values[idx] = m_matrix.values[it];

				if (ks_col[l] < j - l)
					ks_col[l] = j - l;
				if (ks_row[j] < l-j)
					ks_row[j] = l-j;
			}
		} // end for
	} else {
		for (int it2 = 0; it2 < m_n; it2++) {
			int start_idx = m_matrix.row_offsets[it2];
			int end_idx = m_matrix.row_offsets[it2+1];

			for (int it = start_idx; it < end_idx; ++it, ++idx) {
				int j = it2;
				int l = m_matrix.column_indices[it];

				Acoo.row_indices[idx] = j;
				Acoo.column_indices[idx] = l;
				Acoo.values[idx] = m_matrix.values[it];

				if (ks_col[l] < j - l)
					ks_col[l] = j - l;
				if (ks_row[j] < l-j)
					ks_row[j] = l-j;
			}
		} // end for
	}

	for (int i=1; i<m_n; i++) {
		if (ks_col[i] < ks_col[i-1] - 1)
			ks_col[i] = ks_col[i-1] - 1;
		if (ks_row[i] < ks_row[i-1] - 1)
			ks_row[i] = ks_row[i-1] - 1;
	}
}

template <typename T>
void
Graph<T>::assembleBandedMatrix(int         bandwidth,
							   bool        saveMem,
                               int         numPartitions,
                               IntVector&  ks_col,
                               IntVector&  ks_row,
							   MatrixCoo&  Acoo,
                               IntVector&  ks,
                               IntVector&  BOffsets,
                               MatrixMap&  typeMap,
                               MatrixMap&  bandedMatMap)
{
	ks.resize(numPartitions, 0);
	BOffsets.resize(numPartitions + 1);

	BOffsets[0] = 0;

	int partSize = m_n / numPartitions;
	int remainder = m_n % numPartitions;
	int factor = (saveMem ? 1 : 2);

	int diagonal_nnz = m_matrix_diagonal.num_entries;

	Acoo.resize(m_n, m_n, diagonal_nnz);

	for (int it2 = 0; it2 < m_n; it2++) {
		int start_idx = m_matrix_diagonal.row_offsets[it2];
		int end_idx = m_matrix_diagonal.row_offsets[it2+1];
		for (int it = start_idx; it < end_idx; ++it) {
			int j = it2;
			int l = m_matrix_diagonal.column_indices[it];
			int curPartNum = l / (partSize + 1);
			if (curPartNum >= remainder)
				curPartNum = remainder + (l-remainder * (partSize + 1)) / partSize;
			if (ks[curPartNum] < abs(l-j))
				ks[curPartNum] = abs(l-j);
		}
	}

	for (int i=0; i < numPartitions; i++) {
		if (i < remainder)
			BOffsets[i+1] = BOffsets[i] + (partSize + 1) * (factor * ks[i] + 1);
		else
			BOffsets[i+1] = BOffsets[i] + (partSize) * (factor * ks[i] + 1);
	}

	if (m_trackReordering) {
		if (typeMap.size() <= 0)
			typeMap.resize(m_nnz);
		bandedMatMap.resize(m_nnz);
	}

	ks_col.resize(m_n, 0);
	ks_row.resize(m_n, 0);

	int idx = 0;
	for (int it2 = 0; it2 < m_n; it2++) {
		int start_idx = m_matrix_diagonal.row_offsets[it2];
		int end_idx = m_matrix_diagonal.row_offsets[it2+1];
		for (int it = start_idx; it < end_idx; ++it, ++idx) {
			int j = it2;
			int l = m_matrix_diagonal.column_indices[it];

			int curPartNum = l / (partSize + 1);
			int l_in_part;
			if (curPartNum >= remainder) {
				l_in_part = l - remainder * (partSize + 1);
				curPartNum = remainder + l_in_part / partSize;
				l_in_part %= partSize;
			} else {
				l_in_part = l % (partSize + 1);
			}

			int K = ks[curPartNum];
			int delta = (saveMem ? 0 : K);
			int i = BOffsets[curPartNum] + l_in_part * (factor * K + 1) + delta + j - l;

			Acoo.row_indices[idx] = j;
			Acoo.column_indices[idx] = l;
			Acoo.values[idx] = m_matrix_diagonal.values[it];

			if (ks_col[l] < j - l)
				ks_col[l] = j - l;
			if (ks_row[j] < l-j)
				ks_row[j] = l-j;

			if (m_trackReordering) {
				int ori_idx = m_ori_indices_diagonal[it];
				typeMap[ori_idx] = 1;
				bandedMatMap[ori_idx] = i;
			}
		}
	}

	int partBegin = 0, partEnd = partSize;
	for (int i=0; i<numPartitions; i++) {
		if (i < remainder)
			partEnd++;
		for (int j = partBegin+1; j < partEnd; j++) {
			if (ks_col[j] < ks_col[j-1] - 1)
				ks_col[j] = ks_col[j-1] - 1;
			if (ks_col[j] > partEnd - j - 1)
				ks_col[j] = partEnd - j - 1;

			if (ks_row[j] < ks_row[j-1] - 1)
				ks_row[j] = ks_row[j-1] - 1;
			if (ks_row[j] > partEnd - j - 1)
				ks_row[j] = partEnd - j - 1;
		}
		partBegin = partEnd;
		partEnd = partBegin + partSize;
	}
}


// ----------------------------------------------------------------------------
// Graph::DB()
//
// This function performs the DB reordering algorithm...
// ----------------------------------------------------------------------------
template <typename T>
bool
Graph<T>::DB(const MatrixCsr& Acsr,
			 bool             scale,
			 bool             dbFirstStageOnly,
             IntVectorD&      d_dbRowPerm,
             DoubleVectorD&   d_dbRowScale,
             DoubleVectorD&   d_dbColScale,
             MatrixMapF&      scaleMap)
{
	CPUTimer loc_timer;

	loc_timer.Start();
	// Allocate space for the output vectors.
	d_dbRowPerm.resize(m_n);

	if (m_trackReordering)
		m_ori_indices.resize(m_nnz);

	// Allocate space for temporary vectors.
	DoubleVectorD  d_c_val(m_nnz);
	DoubleVectorD  d_max_val_in_col(m_n, 0);

	BoolVector     matched(m_n, 0);
	BoolVector     rev_matched(m_n, 0);
	IntVector      dbRowReordering(m_n, 0);
	IntVector      rev_match_nodes(m_nnz);

	get_csc_matrix(Acsr, d_c_val, d_max_val_in_col);
	loc_timer.Stop();
	m_timeDB_pre = loc_timer.getElapsed();

	loc_timer.Start();
	DoubleVector c_val           = d_c_val;
	DoubleVector dbRowScale(m_n);
	DoubleVector dbColScale(m_n);
	init_reduced_cval(dbFirstStageOnly, Acsr.row_offsets, Acsr.column_indices, c_val, dbColScale, dbRowScale, dbRowReordering, rev_match_nodes, matched, rev_matched);
	loc_timer.Stop();
	m_timeDB_first = loc_timer.getElapsed();

	loc_timer.Start();

	{
		IntVector  irn(m_n);
		IntVector  prev(m_n);
		for(int i=0; i<m_n; i++) {
			if(rev_matched[i]) continue;
			find_shortest_aug_path(i, matched, rev_matched, dbRowReordering, rev_match_nodes, Acsr.row_offsets, Acsr.column_indices, prev, dbColScale, dbRowScale, c_val, irn);
		}

		{
			for (int i=0; i<m_n; i++)
				if (!matched[i])
					throw system_error(system_error::Matrix_singular, "Singular matrix found");
		}

		DoubleVector max_val_in_col = d_max_val_in_col;
		thrust::transform(dbColScale.begin(), dbColScale.end(), dbColScale.begin(), Exponential());
		thrust::transform(thrust::make_transform_iterator(dbRowScale.begin(), Exponential()),
				thrust::make_transform_iterator(dbRowScale.end(), Exponential()),
				max_val_in_col.begin(),
				dbRowScale.begin(),
				thrust::divides<double>());


		d_dbRowScale = dbRowScale;
		d_dbColScale = dbColScale;
	}
	loc_timer.Stop();
	m_timeDB_second = loc_timer.getElapsed();

	IntVectorD d_dbRowReordering   =  dbRowReordering;
	thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(m_n), d_dbRowReordering.begin(), d_dbRowPerm.begin());


	loc_timer.Start();

	if (m_trackReordering)
		scaleMap.resize(m_nnz, T(1.0));

	// TODO: how to do scale when we apply only the first stage
	if (dbFirstStageOnly)
		scale = false;

	IntVector dbRowPerm = d_dbRowPerm;
	IntVector row_indices(m_nnz);
	// m_matrix  = Acsr;
	m_matrix.resize(m_n, m_n, m_nnz);
	thrust::copy(Acsr.column_indices.begin(), Acsr.column_indices.end(), m_matrix.column_indices.begin());
	if (scale) {
		for (int i = 0; i < m_n; i++) {
			int start_idx = Acsr.row_offsets[i], end_idx = Acsr.row_offsets[i+1];
			int new_row = dbRowPerm[i];
			for (int l = start_idx; l < end_idx; l++) {
				row_indices[l] = new_row;
				int to   = (Acsr.column_indices[l]);
				T scaleFact = (T)(dbRowScale[i] * dbColScale[to]);
				m_matrix.values[l] = scaleFact * Acsr.values[l];

				if (m_trackReordering)
					scaleMap[l] = scaleFact;
			}
		}
	} else {
		for (int i = 0; i < m_n; i++) {
			int start_idx = Acsr.row_offsets[i], end_idx = Acsr.row_offsets[i+1];
			int new_row = dbRowPerm[i];
			for (int l = start_idx; l < end_idx; l++) {
				row_indices[l] = new_row;
				m_matrix.values[l] = Acsr.values[l];
			}
		}

		if (m_trackReordering)
			cusp::blas::fill(scaleMap, (T) 1.0);
	}

	/*
	{
		thrust::sort_by_key(row_indices.begin(), row_indices.end(), thrust::make_zip_iterator(thrust::make_tuple(m_matrix.column_indices.begin(), m_matrix.values.begin())));
		cusp::indices_to_offsets(row_indices, m_matrix.row_offsets);
	} */
	{
		IntVector& row_offsets = m_matrix.row_offsets;
		int&       nnz         = m_nnz;
		thrust::fill(row_offsets.begin(), row_offsets.end(), 0);
		IntVector column_indices(nnz);
		Vector    values(nnz);
		for (int i = 0; i < nnz; i++)
			row_offsets[row_indices[i]] ++;

		thrust::inclusive_scan(row_offsets.begin(), row_offsets.end(), row_offsets.begin());

		for (int i = nnz - 1; i >= 0; i--) {
			int idx = (--row_offsets[row_indices[i]]);
			column_indices[idx] = m_matrix.column_indices[i];
			values[idx] = m_matrix.values[i];
			if (m_trackReordering)
				m_ori_indices[idx] = i;
		}
		m_matrix.column_indices = column_indices;
		m_matrix.values         = values;
	}
	loc_timer.Stop();
	m_timeDB_post = loc_timer.getElapsed();

	return true;
}


// ----------------------------------------------------------------------------
// Graph::RCM()
//
// This function implements the Reverse Cuthill-McKee algorithm...
// The return value is the obtained bandwidth. A value of -1 is returned if
// the algorithm fails.
// ----------------------------------------------------------------------------
template <typename T>
int
Graph<T>::RCM(MatrixCsr&   mat_csr,
              IntVector&   optReordering,
              IntVector&   optPerm)
{
	optReordering.resize(m_n);
	optPerm.resize(m_n);

	int nnz = mat_csr.num_entries;

	IntVector tmp_reordering(m_n);

	thrust::sequence(optReordering.begin(), optReordering.end());

	IntVector row_indices(nnz);
	IntVector column_indices(nnz);
	IntVector row_offsets(m_n + 1);
	IntVector ori_degrees(m_n);
#ifdef USE_OLD_CUSP
	cusp::detail::offsets_to_indices(mat_csr.row_offsets, row_indices);
#else
	cusp::offsets_to_indices(mat_csr.row_offsets, row_indices);
#endif
	thrust::transform(mat_csr.row_offsets.begin() + 1, mat_csr.row_offsets.end(), mat_csr.row_offsets.begin(), ori_degrees.begin(), thrust::minus<int>());

	EdgeIterator begin = thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), mat_csr.column_indices.begin()));
	EdgeIterator end   = thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   mat_csr.column_indices.end()));
	int tmp_bdwidth = thrust::transform_reduce(begin, end, EdgeLength(), 0, thrust::maximum<int>());
	int bandwidth = tmp_bdwidth;
	buildTopology(begin, end, 0, m_n, row_offsets, column_indices);

	const int MAX_NUM_TRIAL = 5;
	const int BANDWIDTH_THRESHOLD = 256;

	CPUTimer timer;
	timer.Start();

	BoolVector tried(m_n, false);
	IntVector pushed(m_n, -1);
	IntVector levels(m_n);

	int max_level = 0;
	int p_max_level = 0;

	for (int trial_num = 0; trial_num < MAX_NUM_TRIAL ; trial_num++)
	{
		std::queue<int> q;
		std::priority_queue<NodeType, std::vector<NodeType>, CompareValue<int> > pq;

		int tmp_node;

		int left_cnt = m_n;
		int j = 0, last = 0;

		if (trial_num > 0) {
			IntIterator max_level_iter = thrust::max_element(levels.begin(), levels.end());
			int max_count = thrust::count(levels.begin(), levels.end(), max_level);

			if (max_count > 1) {
				IntVector max_level_vertices(max_count);
				IntVector max_level_valence(max_count);

				thrust::copy_if(thrust::counting_iterator<int>(0),
						thrust::counting_iterator<int>(int(m_n)),
						levels.begin(),
						max_level_vertices.begin(),
						EqualTo<int>(max_level));

				thrust::gather(thrust::counting_iterator<int>(0),
						thrust::counting_iterator<int>(max_count),
						ori_degrees.begin(),
						max_level_valence.begin());

				int min_valence_pos = thrust::min_element(max_level_valence.begin(), max_level_valence.end()) - max_level_valence.begin();
				tmp_node = max_level_vertices[min_valence_pos];
			} else
				tmp_node = max_level_iter - levels.begin();

			while(tried[tmp_node])
				tmp_node = (tmp_node + 1) % m_n;
		} else
			tmp_node = thrust::min_element(ori_degrees.begin(), ori_degrees.end()) - ori_degrees.begin();

		tried[tmp_node]  = true;
		levels[tmp_node] = 0;
		pushed[tmp_node] = trial_num;
		q.push(tmp_node);

		while(left_cnt--) {
			if(q.empty()) {
				left_cnt++;
				int i;

				for(i = last; i < m_n; i++) {
					if(pushed[i] != trial_num) {
						q.push(i);
						pushed[i] = trial_num;
						last = i;
						break;
					}
				}
				if(i < m_n) continue;
				fprintf(stderr, "Can never get here!\n");
				return -1;
			}

			tmp_node = q.front();
			tmp_reordering[j] = tmp_node;
			j++;

			q.pop();

			int start_idx = row_offsets[tmp_node], end_idx = row_offsets[tmp_node + 1];
			int local_level = levels[tmp_node];

			for (int i = start_idx; i < end_idx; i++)  {
				int target_node = column_indices[i];
				if(pushed[target_node] != trial_num) {
					pushed[target_node] = trial_num;
					pq.push(thrust::make_tuple(target_node, ori_degrees[target_node]));
					levels[target_node] = local_level + 1;
				}
			}

			while(!pq.empty()) {
				q.push(thrust::get<0>(pq.top()));
				pq.pop();
			}
		}

		thrust::scatter(thrust::make_counting_iterator(0), 
						thrust::make_counting_iterator(m_n),
						tmp_reordering.begin(),
						optPerm.begin());

		{
			int *perm_array = thrust::raw_pointer_cast(&optPerm[0]);
			tmp_bdwidth = thrust::transform_reduce(begin, end, PermutedEdgeLength(perm_array), 0, thrust::maximum<int>());
		}

		if(bandwidth > tmp_bdwidth) {
			bandwidth = tmp_bdwidth;
			optReordering = tmp_reordering;
		}

		if (trial_num > 0) {
			if (p_max_level >= max_level)
				break;

			const double stop_ratio = 0.01;
			double max_level_ratio = 1.0 * (max_level - p_max_level) / p_max_level;

			if (max_level_ratio < stop_ratio)
				break;

			p_max_level = max_level;
		} else
			p_max_level = max_level;

		if(bandwidth <= BANDWIDTH_THRESHOLD)
			break;
	}


	timer.Stop();
	m_timeRCM = timer.getElapsed();

	thrust::scatter(thrust::make_counting_iterator(0), 
	                thrust::make_counting_iterator(m_n),
	                optReordering.begin(),
	                optPerm.begin());

	{
		int* perm_array = thrust::raw_pointer_cast(&optPerm[0]);
		thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), mat_csr.column_indices.begin())),
						  thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   mat_csr.column_indices.end())),
						  thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), mat_csr.column_indices.begin())),
						  PermuteEdge(perm_array));

		// thrust::sort_by_key(row_indices.begin(), row_indices.end(), thrust::make_zip_iterator(thrust::make_tuple(mat_csr.column_indices.begin(), mat_csr.values.begin())));
		{
			thrust::fill(row_offsets.begin(), row_offsets.end(), 0);
			IntVector column_indices(nnz);
			Vector    values(nnz);
			IntVector ori_indices;
			if (m_trackReordering)
				ori_indices.resize(nnz);

			for (int i = 0; i < nnz; i++)
				row_offsets[row_indices[i]] ++;

			thrust::inclusive_scan(row_offsets.begin(), row_offsets.end(), row_offsets.begin());

			for (int i = nnz - 1; i >= 0; i--) {
				int idx = (--row_offsets[row_indices[i]]);
				column_indices[idx] = mat_csr.column_indices[i];
				values[idx] = mat_csr.values[i];
				if (m_trackReordering)
					ori_indices[idx] = m_ori_indices[i];
			}

			mat_csr.column_indices = column_indices;
			mat_csr.values         = values;
			mat_csr.row_offsets    = row_offsets;

			if (m_trackReordering)
				m_ori_indices = ori_indices;
		}
		// cusp::indices_to_offsets(row_indices, mat_csr.row_offsets);
	}

	return bandwidth;
}


// ----------------------------------------------------------------------------
// Graph::partitionedRCM()
//
// This function implements the second-level Reverse Cuthill-McKee algorithm,
// dealing with a specified partition
// ----------------------------------------------------------------------------
template <typename T>
bool
Graph<T>::partitionedRCM(MatrixCsr&     mat_csr,
						 int            index_begin,
						 int            index_end,
                         int            node_begin,
                         int            node_end,
                         IntVector&     optReordering,
                         IntVector&     optPerm,
						 IntVector&     row_offsets,
						 IntVector&     row_indices)
{
	thrust::sequence(optReordering.begin()+node_begin, optReordering.begin()+node_end, node_begin);

	int tmp_bdwidth = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin() + index_begin, mat_csr.column_indices.begin() + index_begin)), 
											   thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin() + index_end, mat_csr.column_indices.begin() + index_end)), 
											   EdgeLength(), 0, thrust::maximum<int>());

	int opt_bdwidth = tmp_bdwidth;
	EdgeIterator begin = thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin() + index_begin, mat_csr.column_indices.begin() + index_begin));
	EdgeIterator end   = thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin() + index_end,   mat_csr.column_indices.begin() + index_end));
	IntVector column_indices;
	buildTopology(begin, end, node_begin, node_end, row_offsets, column_indices);

	const int MAX_NUM_TRIAL = 5;
	const int BANDWIDTH_THRESHOLD = 128;

	BoolVector tried(m_n, false);
	IntVector  pushed(node_end, -1);
	IntVector  ori_degrees(node_end - node_begin);
	IntVector  levels(m_n);
	int        max_level, p_max_level;
	thrust::transform(mat_csr.row_offsets.begin() + (node_begin + 1), mat_csr.row_offsets.begin() + (node_end), mat_csr.row_offsets.begin() + node_begin, ori_degrees.begin(), thrust::minus<int>());


	CPUTimer timer;
	timer.Start();

	for (int trial_num = 0; trial_num < MAX_NUM_TRIAL; trial_num++) {
		std::queue<int> q;
		std::priority_queue<NodeType, std::vector<NodeType>, CompareValue<int> > pq;

		int tmp_node;

		if (trial_num > 0) {
			IntIterator max_level_iter = thrust::max_element(levels.begin() + node_begin, levels.begin() + node_end);
			int max_count = thrust::count(levels.begin() + node_begin, levels.begin() + node_end, max_level);

			if (max_count > 1) {
				IntVector max_level_vertices(max_count);
				IntVector max_level_valence(max_count);

				thrust::copy_if(thrust::counting_iterator<int>(node_begin),
						thrust::counting_iterator<int>(node_end),
						levels.begin()+node_begin,
						max_level_vertices.begin(),
						EqualTo<int>(max_level));

				thrust::gather(thrust::counting_iterator<int>(0),
						thrust::counting_iterator<int>(max_count),
						ori_degrees.begin(),
						max_level_valence.begin());

				int min_valence_pos = thrust::min_element(max_level_valence.begin(), max_level_valence.end()) - max_level_valence.begin();
				tmp_node = max_level_vertices[min_valence_pos];
			} else
				tmp_node = max_level_iter - levels.begin();

			while(tried[tmp_node])
				tmp_node = (tmp_node + 1) % m_n;
		} else
			tmp_node = thrust::min_element(ori_degrees.begin(), ori_degrees.end()) - ori_degrees.begin() + node_begin;

		tried[tmp_node]  = true;
		levels[tmp_node] = 0;
		pushed[tmp_node] = trial_num;
		q.push(tmp_node);

		int left_cnt = node_end - node_begin;
		int j = node_begin, last = node_begin;

		while(left_cnt--) {
			if(q.empty()) {
				left_cnt++;
				int i;
				for(i = last; i < node_end; i++) {
					if(pushed[i] != trial_num) {
						q.push(i);
						pushed[i] = trial_num;
						last = i;
						break;
					}
				}
				if(i < node_end) continue;
				fprintf(stderr, "Can never get here!\n");
				return false;
			}

			tmp_node = q.front();
			m_buffer_reordering[j] = tmp_node;
			j++;

			q.pop();

			int start_idx = row_offsets[tmp_node], end_idx = row_offsets[tmp_node + 1];

			for (int i = start_idx; i < end_idx; i++)  {
				int target_node = column_indices[i];
				if(pushed[target_node] != trial_num) {
					pushed[target_node] = trial_num;
					pq.push(thrust::make_tuple(target_node, row_offsets[target_node + 1] - row_offsets[target_node]));
					max_level = levels[target_node] = levels[tmp_node] + 1;
				}
			}

			while(!pq.empty()) {
				q.push(thrust::get<0>(pq.top()));
				pq.pop();
			}
		}

		thrust::scatter(thrust::make_counting_iterator(node_begin),
		                thrust::make_counting_iterator(node_end),
		                m_buffer_reordering.begin() + node_begin,
		                optPerm.begin());

		{
			int *perm_array = thrust::raw_pointer_cast(&optPerm[0]);
			tmp_bdwidth = thrust::transform_reduce(begin, end, PermutedEdgeLength(perm_array), 0, thrust::maximum<int>());
		}

		if(opt_bdwidth > tmp_bdwidth) {
			opt_bdwidth = tmp_bdwidth;

			thrust::copy(m_buffer_reordering.begin()+node_begin, m_buffer_reordering.begin()+node_end, optReordering.begin()+node_begin);
		}

		if (trial_num > 0) {
			if (p_max_level >= max_level)
				break;

			const double stop_ratio = 0.01;
			double max_level_ratio = 1.0 * (max_level - p_max_level) / p_max_level;

			if (max_level_ratio < stop_ratio)
				break;

			p_max_level = max_level;
		} else
			p_max_level = max_level;

		if(opt_bdwidth <= BANDWIDTH_THRESHOLD)
			break;
	}

	timer.Stop();
	m_timeRCM += timer.getElapsed();

	thrust::scatter(thrust::make_counting_iterator(node_begin),
	                thrust::make_counting_iterator(node_end),
	                optReordering.begin() + node_begin,
	                optPerm.begin());

	return true;
}

//	--------------------------------------------------------------------------
//	Graph::buildTopology()
//
//	This function builds the topology for the graph for RCM processing
//	--------------------------------------------------------------------------
template <typename T>
void
Graph<T>::buildTopology(EdgeIterator&      begin,
                        EdgeIterator&      end,
						int                node_begin,
						int                node_end,
                        IntVector&         row_offsets,
                        IntVector&         column_indices)
{
	if (row_offsets.size() != m_n + 1)
		row_offsets.resize(m_n + 1, 0);
	else
		thrust::fill(row_offsets.begin(), row_offsets.end(), 0);

	IntVector row_indices((end - begin) << 1);
	column_indices.resize((end - begin) << 1);
	int actual_cnt = 0;

	for(EdgeIterator edgeIt = begin; edgeIt != end; edgeIt++) {
		int from = thrust::get<0>(*edgeIt), to = thrust::get<1>(*edgeIt);
		if (from != to) {
			row_indices[actual_cnt]        = from;
			column_indices[actual_cnt]     = to;
			row_indices[actual_cnt + 1]    = to;
			column_indices[actual_cnt + 1] = from;
			actual_cnt += 2;
		}
	}
	row_indices.resize(actual_cnt);
	column_indices.resize(actual_cnt);
	// thrust::sort_by_key(row_indices.begin(), row_indices.end(), column_indices.begin());
	{
		int&      nnz = actual_cnt;
		IntVector tmp_column_indices(nnz);
		for (int i = 0; i < nnz; i++)
			row_offsets[row_indices[i]] ++;

		thrust::inclusive_scan(row_offsets.begin() + node_begin, row_offsets.begin() + (node_end + 1), row_offsets.begin() + node_begin);

		for (int i = nnz - 1; i >= 0; i--) {
			int idx = (--row_offsets[row_indices[i]]);
			tmp_column_indices[idx] = column_indices[i];
		}
		column_indices = tmp_column_indices;
	}
}

// ----------------------------------------------------------------------------
// Graph::get_csc_matrix
// Graph::init_reduced_cval
// Graph::find_shortest_aug_path
//
// These are the worker functions for the DB algorithm.
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// Graph::get_csc_matrix()
//
// This function initializes the bipartite graph used in DB. By specially
// assigning weights on edges, DB can vary (this version is option 5 in intel's
// DB).
// ----------------------------------------------------------------------------
template<typename T>
void
Graph<T>::get_csc_matrix(const MatrixCsr&  Acsr,
                         DoubleVectorD&    c_val,
                         DoubleVectorD&    max_val_in_col)
{
	int nnz = Acsr.num_entries;
	IntVectorD    d_row_offsets = Acsr.row_offsets;

	c_val   = Acsr.values;
	{
		IntVectorD    d_row_indices(nnz);
#ifdef USE_OLD_CUSP
		cusp::detail::offsets_to_indices(d_row_offsets, d_row_indices);
#else
		cusp::offsets_to_indices(d_row_offsets, d_row_indices);
#endif
		thrust::transform(c_val.begin(), c_val.end(), c_val.begin(), AbsoluteValue<double>());
		thrust::reduce_by_key(d_row_indices.begin(), d_row_indices.end(), c_val.begin(), thrust::make_discard_iterator(), max_val_in_col.begin(), thrust::equal_to<double>(), thrust::maximum<double>());
	}

	double *dc_val_ptr    = thrust::raw_pointer_cast(&c_val[0]);
	const int *d_row_ptrs = thrust::raw_pointer_cast(&d_row_offsets[0]);
	double *dmax_val_ptr  = thrust::raw_pointer_cast(&max_val_in_col[0]);

	int blockX = m_n, blockY = 1;
	kernelConfigAdjust(blockX, blockY, MAX_GRID_DIMENSION);
	dim3 grids(blockX, blockY);

	device::getResidualValues<<<grids, 64>>>(m_n, dc_val_ptr, dmax_val_ptr, d_row_ptrs); 
}

template<typename T>
void
Graph<T>::get_csr_matrix(MatrixCsr&       Acsr, int numPartitions)
{
	if (numPartitions == 1)
		Acsr = m_matrix;
	else
		Acsr = m_matrix_diagonal;
}

// ----------------------------------------------------------------------------
// Graph::init_reduced_cval()
//
// This function assigns a (partial) match for speeding up DB. If this function
// were not to be called, we should start from zero: no match found at the beginning.
// ----------------------------------------------------------------------------
template <typename T>
void 
Graph<T>::init_reduced_cval(bool           first_stage_only,
							const IntVector&     row_ptr,
                            const IntVector&     rows,
                            DoubleVector&  c_val,
                            DoubleVector&  u_val,
                            DoubleVector&  v_val,
                            IntVector&     match_nodes,
                            IntVector&     rev_match_nodes,
                            BoolVector&    matched,
                            BoolVector&    rev_matched) 
{
	{
		IntVector row_indices(c_val.size());
#ifdef USE_OLD_CUSP
		cusp::detail::offsets_to_indices(row_ptr, row_indices);
#else
		cusp::offsets_to_indices(row_ptr, row_indices);
#endif
		thrust::reduce_by_key(row_indices.begin(), row_indices.end(), c_val.begin(), thrust::make_discard_iterator(), u_val.begin(), thrust::equal_to<double>(), thrust::minimum<double>());
	}


	cusp::blas::fill(v_val, LOC_INFINITY);
	for(int i = 0; i < m_n; i++) {
		int start_idx = row_ptr[i], end_idx = row_ptr[i+1];
		int min_idx = -1;
		for(int j = start_idx; j < end_idx; j++) {
			if (c_val[j] > LOC_INFINITY / 2.0) continue;
			int row = rows[j];
			double tmp_val = c_val[j] - u_val[row];
			if(v_val[i] > tmp_val) {
				v_val[i] = tmp_val;
				min_idx = j;
			}
		}
		if(min_idx >= 0) {
			int tmp_row = rows[min_idx];
			if(!matched[tmp_row]) {
				rev_matched[i] = true;
				matched[tmp_row] = true;
				match_nodes[tmp_row] = i;
				rev_match_nodes[i] = min_idx;
			}
		}
	}

	if (first_stage_only) {
		const double CMP_THRESHOLD = 1e-10;
		for (int i = 0; i < m_n; i++) {
			if (rev_matched[i]) continue;

			int start_idx = row_ptr[i], end_idx = row_ptr[i+1];
			int cur_v_val = v_val[i];

			for (int j = start_idx; j < end_idx; j++) {
				if (c_val[j] > LOC_INFINITY / 2.0) continue;
				int row = rows[j];

				if (!matched[row]) continue;

				double res_cval = c_val[j] - u_val[row] - cur_v_val;

				if (res_cval > CMP_THRESHOLD) continue;

				int col2 = match_nodes[row];
				int start_idx2 = row_ptr[col2], end_idx2 = row_ptr[col2 + 1];

				bool found = false;

				for (int j2 = start_idx2; j2 < end_idx2; j2++) {
					if (c_val[j2] > LOC_INFINITY / 2.0) continue;

					int row2 = rows[j2];
					if (matched[row2]) continue;

					matched[row2] = true;
					match_nodes[row2] = col2;
					rev_match_nodes[col2] = j2;
					match_nodes[row] = i;
					rev_matched[i] = true;
					rev_match_nodes[i] = j;
					found = true;
					break;
				}

				if (found) break;
			}
		}

		for (int i = 0; i < m_n; i++) {
			if (rev_matched[i]) continue;

			int start_idx = row_ptr[i], end_idx = row_ptr[i+1];

			for (int j = start_idx; j < end_idx; j++) {
				int row = rows[j];
				if (!matched[row]) {
					matched[row] = true;
					rev_matched[i] = true;
					match_nodes[row] = i;
					rev_match_nodes[i] = j;
					break;
				}
			}
		}

		for (int i = 0; i < m_n; i++) {
			if (rev_matched[i]) continue;

			int start_idx = row_ptr[i], end_idx = row_ptr[i+1];

			for (int j = start_idx; j < end_idx; j++) {
				int row = rows[j];

				int col2 = match_nodes[row];
				int start_idx2 = row_ptr[col2], end_idx2 = row_ptr[col2+1];

				bool found = false;
				for (int j2 = start_idx2; j2 < end_idx2; j2++) {
					int row2 = rows[j2];

					if (matched[row2]) continue;

					found = true;
					matched[row2] = true;
					match_nodes[row2] = col2;
					rev_match_nodes[col2] = j2;
					match_nodes[row] = i;
					rev_matched[i] = true;
					rev_match_nodes[i] = j;

					break;
				}
				if (found) break;
			}
		}

		int last_j = 0;
		for (int i = 0; i < m_n; i++) {
			if (rev_matched[i]) continue;

			for (int j = last_j; j < m_n; j++) {
				if (matched[j]) continue;
				rev_matched[i] = matched[j] = true;
				match_nodes[j] = i;

				last_j = j + 1;
				break;
			}
		}

		cusp::blas::fill(u_val, 1.0);
		cusp::blas::fill(v_val, 1.0);
	} else {
		thrust::transform_if(u_val.begin(), u_val.end(), matched.begin(), u_val.begin(), ClearValue(), is_not());
		thrust::transform_if(v_val.begin(), v_val.end(), rev_matched.begin(), v_val.begin(), ClearValue(), is_not());
	}

}

// ----------------------------------------------------------------------------
// Graph::find_shortest_aug_path()
//
// The core part of the algorithm of finding minimum match: finding the shortest
// augmenting path and applying it.
// ----------------------------------------------------------------------------
template<typename T>
bool
Graph<T>::find_shortest_aug_path(int            init_node,
                                 BoolVector&    matched,
                                 BoolVector&    rev_matched,
                                 IntVector&     match_nodes,
                                 IntVector&     rev_match_nodes,
                                 const IntVector&     row_ptr,
                                 const IntVector&     rows,
                                 IntVector&     prev,
                                 DoubleVector&  u_val,
                                 DoubleVector&  v_val,
                                 DoubleVector&  c_val,
                                 IntVector&     irn)
{
	bool success = false;

	int b_cnt = 0;

	std::priority_queue<Dijkstra, std::vector<Dijkstra>, CompareValue<double> > Q;

	double lsp = 0.0;
	double lsap = LOC_INFINITY;
	int cur_node = init_node;

	int i;

	int isap = -1;
	int ksap = -1;
	prev[init_node] = -1;

	while(1) {
		int start_cur = row_ptr[cur_node];
		int end_cur = row_ptr[cur_node+1];
		for(i = start_cur; i < end_cur; i++) {
			int cur_row = rows[i];
			if(m_DB_inB[cur_row]) continue;
			if(c_val[i] > LOC_INFINITY / 2.0) continue;
			double reduced_cval = c_val[i] - u_val[cur_row] - v_val[cur_node];
			if (reduced_cval + 1e-10 < 0)
				throw system_error(system_error::Negative_DB_weight, "Negative reduced weight in DB.");
			double d_new = lsp + reduced_cval;
			if(d_new < lsap) {
				if(!matched[cur_row]) {
					lsap = d_new;
					isap = cur_row;
					ksap = i;

					match_nodes[isap] = cur_node;
				} else if (d_new < m_DB_d_vals[cur_row]){
					m_DB_d_vals[cur_row] = d_new;
					prev[match_nodes[cur_row]] = cur_node;
					Q.push(thrust::make_tuple(cur_row, d_new));
					irn[cur_row] = i;
				}
			}
		}

		Dijkstra min_d;
		bool found = false;

		while(!Q.empty()) {
			min_d = Q.top();
			Q.pop();
			if(m_DB_visited[thrust::get<0>(min_d)]) 
				continue;
			found = true;
			break;
		}
		if(!found)
			break;

		int tmp_idx = thrust::get<0>(min_d);
		m_DB_visited[tmp_idx] = true;

		lsp = thrust::get<1>(min_d);
		if(lsap <= lsp) {
			m_DB_visited[tmp_idx] = false;
			m_DB_d_vals[tmp_idx] = LOC_INFINITY;
			break;
		}
		m_DB_inB[tmp_idx] = true;
		m_DB_B[b_cnt++] = tmp_idx;

		cur_node = match_nodes[tmp_idx];
	}

	if(lsap < LOC_INFINITY / 2.0) {
		matched[isap] = true;
		cur_node = match_nodes[isap];

		v_val[cur_node] = c_val[ksap];

		while(prev[cur_node] >= 0) {
			match_nodes[isap] = cur_node;

			int next_ksap = rev_match_nodes[cur_node];
			int next_isap = rows[next_ksap];
			next_ksap = irn[next_isap];

			rev_match_nodes[cur_node] = ksap;

			cur_node = prev[cur_node];
			isap = next_isap;
			ksap = next_ksap;
		}
		match_nodes[isap] = cur_node;
		rev_match_nodes[cur_node] = ksap;
		rev_matched[cur_node] = true;
		success = true;

		for (i = 0; i < b_cnt; i++) {
			int tmp_row = m_DB_B[i];
			int j_val = match_nodes[tmp_row];
			int tmp_k = rev_match_nodes[j_val];
			u_val[tmp_row] += m_DB_d_vals[tmp_row] - lsap;
			v_val[j_val] = c_val[tmp_k] - u_val[tmp_row];
			m_DB_d_vals[tmp_row] = LOC_INFINITY;
			m_DB_visited[tmp_row] = false;
			m_DB_inB[tmp_row] = false;
		}

		while(!Q.empty()) {
			Dijkstra tmpD = Q.top();
			Q.pop();
			m_DB_d_vals[thrust::get<0>(tmpD)] = LOC_INFINITY;
		}
	}

	return success;
}

template<typename T>
int
Graph<T>::sloan(MatrixCsr&   matcsr,
	            IntVector&   optReordering,
	            IntVector&   optPerm)
{
	IntVector   row_indices(m_nnz);
	IntVector   tmp_row_indices(m_nnz << 1);
	IntVector   tmp_column_indices(m_nnz << 1);
	IntVector   tmp_row_offsets(m_n + 1);

	IntVector&  column_indices = matcsr.column_indices;
	IntVector&  row_offsets    = matcsr.row_offsets;

#ifdef USE_OLD_CUSP
	cusp::detail::offsets_to_indices(row_offsets, row_indices);
#else
	cusp::offsets_to_indices(row_offsets, row_indices);
#endif

	EdgeIterator begin = thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), column_indices.begin()));
	EdgeIterator end   = thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   column_indices.end()));
	buildTopology(begin, end, 0, m_n, tmp_row_offsets, tmp_column_indices);

	IntVector   ori_degrees(m_n);
	BoolVector  tried(m_n, false);
	IntVector   visited(m_n, -1);
	IntVector   levels(m_n);

	thrust::transform(tmp_row_offsets.begin() + 1, tmp_row_offsets.end(), tmp_row_offsets.begin(), ori_degrees.begin(), thrust::minus<int>());

	optReordering.resize(m_n);
	optPerm.resize(m_n);

	unorderedBFS(false,
			     true,
				 optReordering,
				 tmp_row_offsets,
				 tmp_column_indices,
				 visited,
				 levels,
				 ori_degrees,
				 tried);

	thrust::scatter(thrust::make_counting_iterator(0),
					thrust::make_counting_iterator(int(m_n)),
					optReordering.begin(),
					optPerm.begin());

	int bandwidth;
	{
		int* perm_array = thrust::raw_pointer_cast(&optPerm[0]);
		thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), matcsr.column_indices.begin())),
				thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   matcsr.column_indices.end())),
				thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), matcsr.column_indices.begin())),
				PermuteEdge(perm_array));

		bandwidth = thrust::transform_reduce(begin, end, EdgeLength(), 0, thrust::maximum<int>());
		{
			int &nnz = m_nnz;
			IntVector row_offsets(m_n + 1, 0);
			IntVector column_indices(nnz);
			Vector    values(nnz);
			IntVector ori_indices;
			if (m_trackReordering)
				ori_indices.resize(nnz);

			for (int i = 0; i < nnz; i++)
				row_offsets[row_indices[i]] ++;

			thrust::inclusive_scan(row_offsets.begin(), row_offsets.end(), row_offsets.begin());

			for (int i = nnz - 1; i >= 0; i--) {
				int idx = (--row_offsets[row_indices[i]]);
				column_indices[idx] = matcsr.column_indices[i];
				values[idx] = matcsr.values[i];
				if (m_trackReordering)
					ori_indices[idx] = m_ori_indices[i];
			}

			matcsr.column_indices = column_indices;
			matcsr.values         = values;
			matcsr.row_offsets    = row_offsets;

			if (m_trackReordering)
				m_ori_indices = ori_indices;
		}
	}

	return bandwidth;
}

template <typename T>
void 
Graph<T>::unorderedBFS(bool          doRCM,
					   bool          doSloan,
					   IntVector&    tmp_reordering,
					   IntVector&    row_offsets,
					   IntVector&    column_indices,
					   IntVector&    visited,
					   IntVector&    levels,
					   IntVector&    ori_degrees,
					   BoolVector&   tried)
{
	int min_idx = thrust::min_element(ori_degrees.begin(), ori_degrees.end()) - ori_degrees.begin();

	visited[min_idx]   = 0;
	tried[min_idx]     = true;
	tmp_reordering[0]  = min_idx;

	int queue_begin = 0, queue_end = 1;

	IntVector comp_offsets(1, 0);

	int last = 0;
	int cur_comp = 0;

	int width = 0, max_width = 0;

	IntVector costs(m_n), ori_costs(m_n);

	StatusVector status(m_n, INACTIVE);

	for (int l = 0; l < m_n; l ++) {
		int n_queue_begin = queue_end;
		if (n_queue_begin - queue_begin > 0) {
			if (width < n_queue_begin - queue_begin)
				width = n_queue_begin - queue_begin;

			for (int l2 = queue_begin; l2 < n_queue_begin; l2 ++) {
				levels[l2] = l;
				int  row = tmp_reordering[l2];
				int  start_idx = row_offsets[row], end_idx = row_offsets[row + 1];

				for (int j = start_idx; j < end_idx; j++) {
					int column = column_indices[j];
					if (visited[column] != 0) {
						visited[column] = 0;
						tmp_reordering[queue_end ++] = column;
					}
				}
			}

			queue_begin = n_queue_begin;
		} else {
			comp_offsets.push_back(queue_begin);
			cur_comp ++;

			if (max_width < width) max_width = width;

			if (queue_begin - comp_offsets[cur_comp - 1] > 32) {
				unorderedBFSIteration(width,
						comp_offsets[cur_comp-1],
						comp_offsets[cur_comp],
						tmp_reordering,
						levels,
						visited,
						row_offsets,
						column_indices,
						ori_degrees,
						tried,
						costs,
						ori_costs,
						status,
						l);
			}
			width = 0;

			if (queue_begin >= m_n) break;

			for (int j = last; j < m_n; j++)
				if (visited[j] < 0) {
					visited[j] = 0;
					tmp_reordering[n_queue_begin] = j;
					last = j;
					tried[j] = true;
					queue_end ++;
					l --;
					break;
				}
		}
	}
}

template <typename T>
void
Graph<T>::unorderedBFSIteration(int            width,
							    int            start_idx,
							    int            end_idx,
							    IntVector&     tmp_reordering,
							    IntVector&     levels,
							    IntVector&     visited,
							    IntVector&     row_offsets,
							    IntVector&     column_indices,
							    IntVector&     ori_degrees,
							    BoolVector&    tried,
							    IntVector&     costs,
							    IntVector&     ori_costs,
							    StatusVector&  status,
							    int &          next_level)
{
	int S = tmp_reordering[start_idx], E = -1;
	int pS = S, pE;

	int next_level_bak = next_level;

	const int ITER_COUNT = 10;

	int p_max_level = levels[end_idx - 1];
	int max_level = p_max_level;
	int start_level = levels[start_idx];

	IntVector tmp_reordering_bak(end_idx - start_idx);

	for (int i = 1; i < ITER_COUNT; i++)
	{
		int max_level_start_idx = thrust::lower_bound(levels.begin() + start_idx, levels.begin() + end_idx, max_level) - levels.begin();

		int max_count = end_idx - max_level_start_idx;

		IntVector max_level_valence(max_count);
		if( max_count > 1 ) {

			thrust::gather(tmp_reordering.begin() + max_level_start_idx, tmp_reordering.begin() + end_idx, ori_degrees.begin(), max_level_valence.begin());

			thrust::sort_by_key(max_level_valence.begin(), max_level_valence.end(), tmp_reordering.begin() + max_level_start_idx);

			E = tmp_reordering[max_level_start_idx];
		}
		else
			E = tmp_reordering[end_idx - 1];

		if (tried[E]) {
			int j;
			for (j = max_level_start_idx; j < end_idx; j++)
				if (!tried[tmp_reordering[j]]) {
					E = tmp_reordering[j];
					break;
				}
			if (j >= end_idx) {
				E = pE;
				S = pS;
				break;
			}
		}
		pE = E;

		int queue_begin = start_idx;
		int queue_end   = start_idx + 1;

		tmp_reordering[start_idx] = E;
		tried[E] = true;
		visited[E] = i;
		levels[start_idx]  = start_level;

		int l;
		int tmp_width = 0;
		for (l = start_level; l < m_n; l ++) {
			int n_queue_begin = queue_end;
			if (tmp_width < n_queue_begin - queue_begin)
				tmp_width = n_queue_begin - queue_begin;

			if (n_queue_begin - queue_begin > 0)
			{
				for (int l2 = queue_begin; l2 < n_queue_begin; l2++) {
					levels[l2] = l;
					int row = tmp_reordering[l2];
					int start_idx = row_offsets[row], end_idx = row_offsets[row + 1];
					for (int j = start_idx; j < end_idx; j++) {
						int column = column_indices[j];
						if (visited[column] != i) {
							visited[column] = i;
							tmp_reordering[queue_end++] = column;
						}
					}
				}
				queue_begin = n_queue_begin;
			} else
				break;
		}

		if (tmp_width > width) {
			next_level = next_level_bak;
			break;
		}

		max_level = levels[end_idx - 1];
		if (max_level <= p_max_level) {
			next_level = max_level + 1;

			break;
		}

		width = tmp_width;


		p_max_level = max_level;
		next_level_bak = next_level = l;

		pS = S;
		S  = E;
	}
	//// thrust::copy(tmp_reordering.begin() + start_idx, tmp_reordering.begin() + end_idx, tmp_reordering_bak.begin());

	const int W1 = 2, W2 = 1;

	for (int i = start_idx; i < end_idx; i++)
		costs[i] = (m_n - 1 - ori_degrees[tmp_reordering[i]]) * W1 + levels[i] * W2;

	thrust::scatter(costs.begin() + start_idx, costs.begin() + end_idx, tmp_reordering.begin() + start_idx, ori_costs.begin());

	//// int head = 0, tail = 1;
	//// tmp_reordering_bak[0] = S;
	status[S] = PREACTIVE;

	int cur_idx = start_idx;

	std::priority_queue<thrust::tuple<int, int > > pq;
	pq.push(thrust::make_tuple(ori_costs[S],S));

	//// while(head < tail) {
	while(! pq.empty()) {
		//// int cur_node = tmp_reordering_bak[head];
		thrust::tuple<int, int> tmp_tuple = pq.top();
		pq.pop();
		int cur_node = thrust::get<1>(tmp_tuple);
		//// int max_cost = ori_costs[cur_node];
		//// int idx = head;
		bool found = (status[cur_node] != POSTACTIVE);
		while (!found) {
			if (pq.empty()) break;
			tmp_tuple = pq.top();
			pq.pop();
			cur_node = thrust::get<1>(tmp_tuple);
			found = (status[cur_node] != POSTACTIVE);
		}

		if (!found) break;

		//// {
			////for (int i = head + 1; i < tail; i++)
	////			if (max_cost < ori_costs[tmp_reordering_bak[i]]) {
	////				idx = i;
	////				cur_node = tmp_reordering_bak[i];
	////				max_cost = ori_costs[cur_node + start_idx];
	////			}
	////	}

		//// if (idx != head) 
			//// tmp_reordering_bak[idx] = tmp_reordering_bak[head];

		if (status[cur_node] == PREACTIVE) {
			int start_idx2 = row_offsets[cur_node], end_idx2 = row_offsets[cur_node + 1];

			for (int l = start_idx2; l < end_idx2; l++) {
				int column = column_indices[l];
				if (status[column] == POSTACTIVE) continue;
				ori_costs[column] += W1;
				pq.push(thrust::make_tuple(ori_costs[column], column));
				if (status[column] == INACTIVE) {
					//// tmp_reordering_bak[tail] = column;
					status[column] = PREACTIVE;
					//// tail ++;
				}
			}
		}

		status[cur_node] = POSTACTIVE;
		tmp_reordering[cur_idx ++] = cur_node;

		int start_idx2 = row_offsets[cur_node], end_idx2 = row_offsets[cur_node + 1];

		for (int l = start_idx2; l < end_idx2; l++) {
			int column = column_indices[l];
			if (status[column] != PREACTIVE) continue;
			ori_costs[column] += W1;
			status[column] = ACTIVE;
			pq.push(thrust::make_tuple(ori_costs[column], column));

			int start_idx3 = row_offsets[column], end_idx3 = row_offsets[column + 1];

			for (int l2 = start_idx3; l2 < end_idx3; l2++) {
				int column2 = column_indices[l2];
				if (status[column2] == POSTACTIVE) continue;

				ori_costs[column2] += W1;
				pq.push(thrust::make_tuple(ori_costs[column2], column2));
				if (status[column2] == INACTIVE) {
					status[column2] = PREACTIVE;
					//// tmp_reordering_bak[tail] = column2;
					//// tail ++;
				}
			}
		}
		//// head++;
	}
}

template <typename T>
size_t
Graph<T>::symbolicFactorization(const MatrixCsr&  Acsr)
{
	std::vector<IntVector> prL(Acsr.num_rows), prU(Acsr.num_rows);

	IntVector A_column_offsets(Acsr.num_rows + 1), A_row_indices(Acsr.num_entries), A_column_indices = Acsr.column_indices;
	{
#ifdef USE_OLD_CUSP
		cusp::detail::offsets_to_indices(Acsr.row_offsets, A_row_indices);
#else
		cusp::offsets_to_indices(Acsr.row_offsets, A_row_indices);
#endif
		thrust::sort_by_key(A_column_indices.begin(), A_column_indices.end(), A_row_indices.begin());
#ifdef USE_OLD_CUSP
		cusp::detail::indices_to_offsets(A_column_indices, A_column_offsets);
#else
		cusp::indices_to_offsets(A_column_indices, A_column_offsets);
#endif
	}

	IntVector L_column_offsets(Acsr.num_rows + 1, 0), L_row_indices;
	IntVector U_row_offsets(Acsr.num_rows + 1, 0),    U_column_indices;
	IntVector pushed_L(Acsr.num_rows, -1);
	IntVector pushed_U(Acsr.num_rows, -1);

	for (int i = 0; i < Acsr.num_rows; i++) {
		int start_idx = A_column_offsets[i], end_idx = A_column_offsets[i+1];
		int l_cur_idx = L_row_indices.size(), u_cur_idx = U_column_indices.size();

		for (int j = start_idx; j < end_idx; j++) {
			int row = A_row_indices[j];
			if (row <= i || pushed_L[row] == i) continue;
			L_row_indices.push_back(row);
			pushed_L[row] = i;
		}

		start_idx = Acsr.row_offsets[i];
		end_idx = Acsr.row_offsets[i+1];

		for (int j = start_idx; j < end_idx; j++) {
			int column = Acsr.column_indices[j];
			if (column <= i  || pushed_U[column] == i) continue;
			U_column_indices.push_back(column);
			pushed_U[column] = i;
		}

		int sizeL = prL[i].size(), sizeU = prU[i].size();

		for (int l = 0; l < sizeL; l++) {
			int column = prL[i][l];
			start_idx = L_column_offsets[column];
			end_idx = L_column_offsets[column + 1];

			for (int j = end_idx - 1; j >= start_idx; j--) {
				int row = L_row_indices[j];
				if (row <= i) break;
				if (pushed_L[row] == i) continue;
				L_row_indices.push_back(row);
				pushed_L[row] = i;
			}
		}

		for (int l = 0; l < sizeU; l++) {
			int row = prU[i][l];
			start_idx = U_row_offsets[row];
			end_idx = U_row_offsets[row+1];

			for (int j = end_idx - 1; j >= start_idx; j--) {
				int column = U_column_indices[j];
				if (column <= i) break;
				if (pushed_U[column] == i) continue;
				U_column_indices.push_back(column);
				pushed_U[column] = i;
			}
		}

		thrust::sort(L_row_indices.begin() + l_cur_idx, L_row_indices.end());
		thrust::sort(U_column_indices.begin() + u_cur_idx, U_column_indices.end());

		int K = Acsr.num_rows - 1;

		bool found = false;
		while (l_cur_idx < L_row_indices.size() && u_cur_idx < U_column_indices.size()) {
			while (l_cur_idx < L_row_indices.size()) {
				if (L_row_indices[l_cur_idx] < U_column_indices[u_cur_idx])
					l_cur_idx ++;
				else if (L_row_indices[l_cur_idx] > U_column_indices[u_cur_idx])
					break;
				else {
					found = true;
					K     = L_row_indices[l_cur_idx];
					break;
				}
			}

			if (found) break;

			while(u_cur_idx < U_column_indices.size()) {
				if (L_row_indices[l_cur_idx] > U_column_indices[u_cur_idx])
					u_cur_idx ++;
				else if (L_row_indices[l_cur_idx] < U_column_indices[u_cur_idx])
					break;
				else {
					found = true;
					K     = L_row_indices[l_cur_idx];
					break;
				}
			}
			if (found) break;
		}
		
		L_column_offsets[i + 1] = L_row_indices.size();
		U_row_offsets[i + 1]    = U_column_indices.size();

		start_idx = L_column_offsets[i];
		end_idx = L_column_offsets[i + 1];
		for (int j = start_idx; j < end_idx; j++) {
			int row = L_row_indices[j];
			if (row > K) break;
			if (row <= i) continue;
			prL[row].push_back(i);
		}

		start_idx = U_row_offsets[i];
		end_idx = U_row_offsets[i + 1];
		for (int j = start_idx; j < end_idx; j++) {
			int column = U_column_indices[j];
			if (column > K) break;
			if (column <= i) continue;
			prU[column].push_back(i);
		}
	}

	return L_row_indices.size() + U_column_indices.size() + (size_t)Acsr.num_rows;
}	


} // namespace sap


#endif
