#ifndef SPIKE_GRAPH_H
#define SPIKE_GRAPH_H

#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <algorithm>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/blas.h>
#include <cusp/print.h>

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/gather.h>

#include <spike/common.h>
#include <spike/timer.h>
#include <spike/device/data_transfer.cuh>
#include <spike/device/mc64.cuh>

#include <spike/exception.h>

namespace spike {

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

	Graph(bool trackReordering = false);

	double     getTimeMC64() const     {return m_timeMC64;}
	double     getTimeMC64Pre() const     {return m_timeMC64_pre;}
	double     getTimeMC64First() const     {return m_timeMC64_first;}
	double     getTimeMC64Second() const     {return m_timeMC64_second;}
	double     getTimeMC64Post() const     {return m_timeMC64_post;}
	double     getTimeRCM() const      {return m_timeRCM;}
	double     getTimeDropoff() const  {return m_timeDropoff;}

	int        reorder(const MatrixCsr& Acsr,
	                   bool             testMC64,
	                   bool             doMC64,
					   bool             mc64FirstStageOnly,
	                   bool             scale,
					   bool             doRCM,
	                   IntVector&       optReordering,
	                   IntVector&       optPerm,
	                   IntVectorD&      d_mc64RowPerm,
	                   VectorD&         d_mc64RowScale,
	                   VectorD&         d_mc64ColScale,
	                   MatrixMapF&      scaleMap,
	                   int&             k_mc64);

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

private:
	int           m_n;
	int           m_nnz;
	MatrixCsr     m_matrix;
	MatrixCsr     m_matrix_diagonal;
	IntVector     m_ori_indices;
	IntVector     m_ori_indices_diagonal;

	bool          m_trackReordering;

	double        m_timeMC64;
	double        m_timeMC64_pre;
	double        m_timeMC64_first;
	double        m_timeMC64_second;
	double        m_timeMC64_post;
	double        m_timeRCM;
	double        m_timeDropoff;

	BoolVector    m_exists;

	bool       MC64(const MatrixCsr& Acsr,
			        bool             scale,
					bool             mc64FirstStageOnly,
	                IntVectorD&      mc64RowPerm,
	                DoubleVectorD&   mc64RowScale,
	                DoubleVectorD&   mc64ColScale,
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

	// Functions used in MC64
	void       find_minimum_match(const MatrixCsr& Acsr,
								  bool             scale,
			                      bool             mc64FirstStageOnly,
								  IntVectorD&      mc64RowPerm,
	                              DoubleVectorD&   mc64RowScale,
	                              DoubleVectorD&   mc64ColScale,
								  MatrixMapF&      scaleMap);
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
:	m_timeMC64(0),
	m_timeMC64_pre(0),
	m_timeMC64_first(0),
	m_timeMC64_second(0),
	m_timeMC64_post(0),
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
                  bool              testMC64,
                  bool              doMC64,
				  bool              mc64FirstStageOnly,
                  bool              scale,
				  bool              doRCM,
                  IntVector&        optReordering,
                  IntVector&        optPerm,
                  IntVectorD&       d_mc64RowPerm,
                  VectorD&          d_mc64RowScale,
                  VectorD&          d_mc64ColScale,
                  MatrixMapF&       scaleMap,
                  int&              k_mc64)
{
	m_n = Acsr.num_rows;
	m_nnz = Acsr.num_entries;

	// Apply mc64 algorithm. Note that we must ensure we always work with
	// double precision scale vectors.
	//
	// TODO:  how can we check if the precision of Vector is already
	//        double, so that we can save extra copies.
	if (doMC64) {
		GPUTimer loc_timer;
		loc_timer.Start();
		DoubleVectorD  mc64RowScaleD;
		DoubleVectorD  mc64ColScaleD;
		MC64(Acsr, scale, mc64FirstStageOnly, d_mc64RowPerm, mc64RowScaleD, mc64ColScaleD, scaleMap);
		d_mc64RowScale = mc64RowScaleD;
		d_mc64ColScale = mc64ColScaleD;
		loc_timer.Stop();
		m_timeMC64 = loc_timer.getElapsed();
	} else {
		d_mc64RowScale.resize(m_n);
		d_mc64ColScale.resize(m_n);
		d_mc64RowPerm.resize(m_n);
		scaleMap.resize(m_nnz);

		m_matrix = Acsr;

		thrust::sequence(d_mc64RowPerm.begin(), d_mc64RowPerm.end());
		cusp::blas::fill(d_mc64RowScale, (T) 1.0);
		cusp::blas::fill(d_mc64ColScale, (T) 1.0);
		cusp::blas::fill(scaleMap, (T) 1.0);
	}

	{
		IntVector row_indices(m_nnz);
		cusp::detail::offsets_to_indices(m_matrix.row_offsets, row_indices);
		k_mc64 = thrust::inner_product(row_indices.begin(), row_indices.end(), m_matrix.column_indices.begin(), 0, thrust::maximum<int>(), Difference());
	}

	if (testMC64)
		return k_mc64;

	// Apply reverse Cuthill-McKee algorithm.
	int bandwidth;
	if (doRCM)
		bandwidth = RCM(m_matrix, optReordering, optPerm);
	else {
		bandwidth = k_mc64;
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

		cusp::detail::indices_to_offsets(Acoo.row_indices, m_matrix.row_offsets);
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

					// FIXME: add support for update
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

					// FIXME: add support for update
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
	cusp::detail::offsets_to_indices(m_matrix_diagonal.row_offsets, row_indices);

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
		// cusp::detail::indices_to_offsets(row_indices, m_matrix_diagonal.row_offsets);
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

	// FIXME: add support for update
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

			// FIXME: add support for update
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
// Graph::MC64()
//
// This function performs the mc64 reordering algorithm...
// ----------------------------------------------------------------------------
template <typename T>
bool
Graph<T>::MC64(const MatrixCsr& Acsr,
			   bool             scale,
			   bool             mc64FirstStageOnly,
               IntVectorD&      d_mc64RowPerm,
               DoubleVectorD&   d_mc64RowScale,
               DoubleVectorD&   d_mc64ColScale,
               MatrixMapF&      scaleMap)
{
	find_minimum_match(Acsr, scale, mc64FirstStageOnly, d_mc64RowPerm, d_mc64RowScale, d_mc64ColScale, scaleMap);
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
	cusp::detail::offsets_to_indices(mat_csr.row_offsets, row_indices);
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
		// cusp::detail::indices_to_offsets(row_indices, mat_csr.row_offsets);
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
	static IntVector tmp_reordering(m_n);

	// for(int i = node_begin; i < node_end; i++)
		// optReordering[i] = i;
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
			tmp_reordering[j] = tmp_node;
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
		                tmp_reordering.begin() + node_begin,
		                optPerm.begin());

		{
			int *perm_array = thrust::raw_pointer_cast(&optPerm[0]);
			tmp_bdwidth = thrust::transform_reduce(begin, end, PermutedEdgeLength(perm_array), 0, thrust::maximum<int>());
		}

		if(opt_bdwidth > tmp_bdwidth) {
			opt_bdwidth = tmp_bdwidth;

			thrust::copy(tmp_reordering.begin()+node_begin, tmp_reordering.begin()+node_end, optReordering.begin()+node_begin);
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
// Graph::find_minimum_match()
// Graph::get_csc_matrix
// Graph::init_reduced_cval
// Graph::find_shortest_aug_path
//
// These are the worker functions for the MC64 algorithm.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Graph::find_minimum_match()
//
// This is the entry function of the core part of MC64 algorithm, which reorders
// the matrix by finding the minimum match of a bipartite graph.
// ----------------------------------------------------------------------------
template <typename T>
void
Graph<T>::find_minimum_match(const MatrixCsr& Acsr,
							 bool             scale,
							 bool             mc64FirstStageOnly, 
							 IntVectorD&      d_mc64RowPerm,
                             DoubleVectorD&   d_mc64RowScale,
                             DoubleVectorD&   d_mc64ColScale,
							 MatrixMapF&      scaleMap)
{
	CPUTimer loc_timer;

	loc_timer.Start();
	// Allocate space for the output vectors.
	d_mc64RowPerm.resize(m_n);

	if (m_trackReordering)
		m_ori_indices.resize(m_nnz);

	// Allocate space for temporary vectors.
	DoubleVectorD  d_c_val(m_nnz);
	DoubleVectorD  d_max_val_in_col(m_n, 0);

	BoolVector     matched(m_n, 0);
	BoolVector     rev_matched(m_n, 0);
	IntVector      mc64RowReordering(m_n, 0);
	IntVector      rev_match_nodes(m_nnz);

	get_csc_matrix(Acsr, d_c_val, d_max_val_in_col);
	loc_timer.Stop();
	m_timeMC64_pre = loc_timer.getElapsed();

	loc_timer.Start();
	DoubleVector c_val           = d_c_val;
	DoubleVector mc64RowScale(m_n);
	DoubleVector mc64ColScale(m_n);
	init_reduced_cval(mc64FirstStageOnly, Acsr.row_offsets, Acsr.column_indices, c_val, mc64ColScale, mc64RowScale, mc64RowReordering, rev_match_nodes, matched, rev_matched);
	loc_timer.Stop();
	m_timeMC64_first = loc_timer.getElapsed();

	loc_timer.Start();

	{
		IntVector  irn(m_n);
		IntVector  prev(m_n);
		for(int i=0; i<m_n; i++) {
			if(rev_matched[i]) continue;
			find_shortest_aug_path(i, matched, rev_matched, mc64RowReordering, rev_match_nodes, Acsr.row_offsets, Acsr.column_indices, prev, mc64ColScale, mc64RowScale, c_val, irn);
		}

		{
			for (int i=0; i<m_n; i++)
				if (!matched[i])
					throw system_error(system_error::Matrix_singular, "Singular matrix found");
		}

		DoubleVector max_val_in_col = d_max_val_in_col;
		thrust::transform(mc64ColScale.begin(), mc64ColScale.end(), mc64ColScale.begin(), Exponential());
		thrust::transform(thrust::make_transform_iterator(mc64RowScale.begin(), Exponential()),
				thrust::make_transform_iterator(mc64RowScale.end(), Exponential()),
				max_val_in_col.begin(),
				mc64RowScale.begin(),
				thrust::divides<double>());


		d_mc64RowScale = mc64RowScale;
		d_mc64ColScale = mc64ColScale;
	}
	loc_timer.Stop();
	m_timeMC64_second = loc_timer.getElapsed();

	IntVectorD d_mc64RowReordering   =  mc64RowReordering;
	thrust::scatter(thrust::make_counting_iterator(0), thrust::make_counting_iterator(m_n), d_mc64RowReordering.begin(), d_mc64RowPerm.begin());


	loc_timer.Start();

	if (m_trackReordering)
		scaleMap.resize(m_nnz, T(1.0));

	// TODO: how to do scale when we apply only the first stage
	if (mc64FirstStageOnly)
		scale = false;

	IntVector mc64RowPerm = d_mc64RowPerm;
	IntVector row_indices(m_nnz);
	// m_matrix  = Acsr;
	m_matrix.resize(m_n, m_n, m_nnz);
	thrust::copy(Acsr.column_indices.begin(), Acsr.column_indices.end(), m_matrix.column_indices.begin());
	if (scale) {
		for (int i = 0; i < m_n; i++) {
			int start_idx = Acsr.row_offsets[i], end_idx = Acsr.row_offsets[i+1];
			int new_row = mc64RowPerm[i];
			for (int l = start_idx; l < end_idx; l++) {
				row_indices[l] = new_row;
				int to   = (Acsr.column_indices[l]);
				T scaleFact = (T)(mc64RowScale[i] * mc64ColScale[to]);
				m_matrix.values[l] = scaleFact * Acsr.values[l];

				if (m_trackReordering)
					scaleMap[l] = scaleFact;
			}
		}
	} else {
		for (int i = 0; i < m_n; i++) {
			int start_idx = Acsr.row_offsets[i], end_idx = Acsr.row_offsets[i+1];
			int new_row = mc64RowPerm[i];
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
		cusp::detail::indices_to_offsets(row_indices, m_matrix.row_offsets);
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
	m_timeMC64_post = loc_timer.getElapsed();
}

// ----------------------------------------------------------------------------
// Graph::get_csc_matrix()
//
// This function initializes the bipartite graph used in MC64. By specially
// assigning weights on edges, MC64 can vary (this version is option 5 in intel's
// MC64).
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
		cusp::detail::offsets_to_indices(d_row_offsets, d_row_indices);
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
// This function assigns a (partial) match for speeding up MC64. If this function
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
		cusp::detail::offsets_to_indices(row_ptr, row_indices);
		thrust::reduce_by_key(row_indices.begin(), row_indices.end(), c_val.begin(), thrust::make_discard_iterator(), u_val.begin(), thrust::equal_to<double>(), thrust::minimum<double>());
	}

#if 0
	int *p_row_offsets    =   thrust::raw_pointer_cast(&row_ptr[0]);
	int *p_column_indices =   thrust::raw_pointer_cast(&rows[0]);
	double *p_values      =   thrust::raw_pointer_cast(&c_val[0]);
	double *p_u_values    =   thrust::raw_pointer_cast(&u_val[0]);
	double *p_v_values    =   thrust::raw_pointer_cast(&v_val[0]);
	int *p_matches        =   thrust::raw_pointer_cast(&match_nodes[0]);
	int *p_rev_matches    =   thrust::raw_pointer_cast(&rev_match_nodes[0]);
	bool *p_matched       =   thrust::raw_pointer_cast(&matched[0]);
	bool *p_rev_matched   =   thrust::raw_pointer_cast(&rev_matched[0]);

	int blockX = m_n, blockY = 1;
	kernelConfigAdjust(blockX, blockY, 32768);
	dim3 grids(blockX, blockY);


	device::findInitialMatch<<<grids, 64>>>(m_n, p_row_offsets, p_column_indices, p_values, p_u_values, p_v_values,
			                                p_matches, p_rev_matches, p_matched, p_rev_matched);

	blockX = 1;
	int threadX = m_n;
	kernelConfigAdjust(threadX, blockX, 512);
	device::clearMatchesWithContention<<<blockX, threadX>>> (m_n, p_column_indices, p_matches, p_rev_matches, p_rev_matched);
#endif

#if 1
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
#endif
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

	static IntVector B(m_n, 0);
	int b_cnt = 0;
	static BoolVector inB(m_n, false);

	std::priority_queue<Dijkstra, std::vector<Dijkstra>, CompareValue<double> > Q;

	double lsp = 0.0;
	double lsap = LOC_INFINITY;
	int cur_node = init_node;

	int i;

	int isap = -1;
	int ksap = -1;
	prev[init_node] = -1;

	static DoubleVector d_vals(m_n, LOC_INFINITY);
	static BoolVector visited(m_n, false);

	while(1) {
		int start_cur = row_ptr[cur_node];
		int end_cur = row_ptr[cur_node+1];
		for(i = start_cur; i < end_cur; i++) {
			int cur_row = rows[i];
			if(inB[cur_row]) continue;
			if(c_val[i] > LOC_INFINITY / 2.0) continue;
			double reduced_cval = c_val[i] - u_val[cur_row] - v_val[cur_node];
			if (reduced_cval + 1e-10 < 0)
				throw system_error(system_error::Negative_MC64_weight, "Negative reduced weight in MC64.");
			double d_new = lsp + reduced_cval;
			if(d_new < lsap) {
				if(!matched[cur_row]) {
					lsap = d_new;
					isap = cur_row;
					ksap = i;

					match_nodes[isap] = cur_node;
				} else if (d_new < d_vals[cur_row]){
					d_vals[cur_row] = d_new;
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
			if(visited[thrust::get<0>(min_d)]) 
				continue;
			found = true;
			break;
		}
		if(!found)
			break;

		int tmp_idx = thrust::get<0>(min_d);
		visited[tmp_idx] = true;

		lsp = thrust::get<1>(min_d);
		if(lsap <= lsp) {
			visited[tmp_idx] = false;
			d_vals[tmp_idx] = LOC_INFINITY;
			break;
		}
		inB[tmp_idx] = true;
		B[b_cnt++] = tmp_idx;

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
			int tmp_row = B[i];
			int j_val = match_nodes[tmp_row];
			int tmp_k = rev_match_nodes[j_val];
			u_val[tmp_row] += d_vals[tmp_row] - lsap;
			v_val[j_val] = c_val[tmp_k] - u_val[tmp_row];
			d_vals[tmp_row] = LOC_INFINITY;
			visited[tmp_row] = false;
			inB[tmp_row] = false;
		}

		while(!Q.empty()) {
			Dijkstra tmpD = Q.top();
			Q.pop();
			d_vals[thrust::get<0>(tmpD)] = LOC_INFINITY;
		}
	}

	return success;
}

} // namespace spike


#endif
