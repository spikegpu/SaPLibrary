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

#include <spike/exception.h>

namespace spike {


class Dijkstra
{
public:
	Dijkstra() {}
	Dijkstra(int idx, double val) : m_idx(idx), m_val(val) {}

	friend bool operator<(const Dijkstra& a, const Dijkstra& b) {
		return a.m_val > b.m_val;
	}

	friend bool operator>(const Dijkstra& a, const Dijkstra& b) {
		return a.m_val < b.m_val;
	}

	int       m_idx;
	double    m_val;
};


class Node
{
public:
	Node(int idx, int degree) : m_idx(idx), m_degree(degree) {}

	friend bool operator>(const Node& a, const Node& b) {
		return a.m_degree < b.m_degree;
	}

	friend bool operator<(const Node& a, const Node& b) {
		return a.m_degree > b.m_degree;
	}

	int  m_idx;
	int  m_degree;
};


template <typename T>
class EdgeT
{
public:
	EdgeT() {}
	EdgeT(int from, int to, T val) : m_from(from), m_to(to), m_val(val) {}

	EdgeT(int ori_idx, int from, int to, T val)
		: m_ori_idx(ori_idx), m_from(from), m_to(to), m_val(val) {}

	friend bool operator< (const EdgeT& a, const EdgeT& b) {
		return a.m_to < b.m_to;
	}

	int  m_from;
	int  m_to;
	T    m_val;

	int m_ori_idx;
};


template <typename T>
class Graph
{
public:
	typedef typename cusp::coo_matrix<int, T, cusp::host_memory> MatrixCoo;
	typedef typename cusp::array1d<T, cusp::host_memory>         Vector;
	typedef typename cusp::array1d<double, cusp::host_memory>    DoubleVector;
	typedef typename cusp::array1d<int, cusp::host_memory>       IntVector;
	typedef typename cusp::array1d<int, cusp::device_memory>     IntVectorD;
	typedef typename cusp::array1d<double, cusp::device_memory>  DoubleVectorD;
	typedef typename cusp::array1d<bool, cusp::host_memory>      BoolVector;
	typedef Vector                                               MatrixMapF;
	typedef IntVector                                            MatrixMap;

	typedef Node                  NodeType;
	typedef EdgeT<T>              EdgeType;
	typedef std::vector<NodeType> NodeVector;
	typedef std::vector<EdgeType> EdgeVector;

	typedef typename EdgeVector::iterator         EdgeIterator;
	typedef typename EdgeVector::reverse_iterator EdgeRevIterator;

	Graph(bool trackReordering = false);

	double     getTimeMC64() const     {return m_timeMC64;}
	double     getTimeMC64Pre() const     {return m_timeMC64_pre;}
	double     getTimeMC64First() const     {return m_timeMC64_first;}
	double     getTimeMC64Second() const     {return m_timeMC64_second;}
	double     getTimeMC64Post() const     {return m_timeMC64_post;}
	double     getTimeRCM() const      {return m_timeRCM;}
	double     getTimeDropoff() const  {return m_timeDropoff;}

	int        reorder(const MatrixCoo& Acoo,
	                   bool             testMC64,
	                   bool             doMC64,
					   bool             mc64FirstStageOnly,
	                   bool             scale,
	                   IntVector&       optReordering,
	                   IntVector&       optPerm,
	                   IntVector&       mc64RowPerm,
	                   Vector&          mc64RowScale,
	                   Vector&          mc64ColScale,
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

private:
	int           m_n;
	int           m_nnz;
	EdgeVector    m_edges;
	EdgeVector    m_major_edges;

	EdgeIterator  m_first;

	bool          m_trackReordering;

	double        m_timeMC64;
	double        m_timeMC64_pre;
	double        m_timeMC64_first;
	double        m_timeMC64_second;
	double        m_timeMC64_post;
	double        m_timeRCM;
	double        m_timeDropoff;

	BoolVector    m_exists;

	bool       MC64(const MatrixCoo& Acoo,
			        bool             scale,
					bool             mc64FirstStageOnly,
	                IntVector&       mc64RowPerm,
	                DoubleVector&    mc64RowScale,
	                DoubleVector&    mc64ColScale,
	                MatrixMapF&      scaleMap);

	int        RCM(EdgeVector&  edges,
	               IntVector&   optReordering,
	               IntVector&   optPerm);

	bool       partitionedRCM(EdgeIterator&  begin,
	                          EdgeIterator&  end,
	                          int            node_begin,
	                          int            node_end,
	                          IntVector&     optReordering,
	                          IntVector&     optPerm);

	void       buildTopology(EdgeIterator&      begin,
	                         EdgeIterator&      end,
	                         IntVector&         degrees,
	                         std::vector<int>*  in_out_graph);

	static const double LOC_INFINITY;

	// Functions used in MC64
	void       find_minimum_match(const MatrixCoo& Acoo,
			                      bool             mc64FirstStageOnly,
								  IntVector&       mc64RowPerm,
	                              DoubleVector&    mc64RowScale,
	                              DoubleVector&    mc64ColScale);
	void       init_reduced_cval(bool           first_stage_only,
			                     IntVector&     row_ptr,
	                             IntVector&     rows,
	                             DoubleVector&  c_val, 
	                             DoubleVector&  u_val,
	                             DoubleVector&  v_val,
	                             IntVector&     match_nodes,
	                             IntVector&     rev_match_nodes,
	                             IntVector&     matched,
	                             IntVector&     rev_matched);
	bool       find_shortest_aug_path(int init_node,
	                                  IntVector& matched, IntVector& rev_matched, 
	                                  IntVector& match_nodes, IntVector& rev_match_nodes,
	                                  IntVector& row_ptr, IntVector& rows, IntVector& prev,
	                                  DoubleVector& u_val,
	                                  DoubleVector& v_val,
	                                  DoubleVector& c_val,
	                                  IntVector&    irn);
	void       get_csc_matrix(const MatrixCoo& Acoo,
							  IntVector&       row_ptr,
	                          IntVector&       rows, 
	                          DoubleVector&    c_val,
	                          DoubleVector&    max_val_in_col);

	// Functor objects.
	struct CompareEdgeLength
	{
		bool operator()(const EdgeT<T>& a, const EdgeT<T>& b)
		{
			return abs(a.m_from - a.m_to) > abs(b.m_from - b.m_to);
		}
	};

	struct AccumulateEdgeValue
	{
		T operator()(T res, const EdgeT<T>& edge)
		{
			return res + std::abs(edge.m_val);
		}
	};

	struct EdgeLength : public thrust::unary_function<EdgeT<T>, int>
	{
		__host__ __device__
		int operator() (const EdgeT<T>& a)
		{
			return std::abs(a.m_from - a.m_to);
		}
	};

	struct PermutedEdgeLength : public thrust::unary_function<EdgeT<T>, int>
	{
		int* m_perm;

		PermutedEdgeLength(int* perm): m_perm(perm) {}

		__host__ __device__
		int operator() (const EdgeT<T>& a)
		{
			return std::abs(m_perm[a.m_from] - m_perm[a.m_to]);
		}
	};

	struct PermuteEdge
	{
		int* m_perm;

		PermuteEdge(int* perm): m_perm(perm) {}

		__host__ __device__
		void operator() (EdgeT<T>& a)
		{
			a.m_from = m_perm[a.m_from];
			a.m_to = m_perm[a.m_to];
		}
	};

	struct Exponential: public thrust::unary_function<double, double>
	{
		__host__
		double operator() (double a)
		{
			return exp(a);
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
Graph<T>::reorder(const MatrixCoo&  Acoo,
                  bool              testMC64,
                  bool              doMC64,
				  bool              mc64FirstStageOnly,
                  bool              scale,
                  IntVector&        optReordering,
                  IntVector&        optPerm,
                  IntVector&        mc64RowPerm,
                  Vector&           mc64RowScale,
                  Vector&           mc64ColScale,
                  MatrixMapF&       scaleMap,
                  int&              k_mc64)
{
	m_n = Acoo.num_rows;
	m_nnz = Acoo.num_entries;

	// Create the edges in the graph.
	if (m_trackReordering) {
		for (int i = 0; i < m_nnz; i++) {
			m_edges.push_back(EdgeType(i, Acoo.row_indices[i], Acoo.column_indices[i], (T)Acoo.values[i]));
		}
	} else {
		for (int i = 0; i < m_nnz; i++)
			m_edges.push_back(EdgeType(Acoo.row_indices[i], Acoo.column_indices[i], (T)Acoo.values[i]));
	}

	// Apply mc64 algorithm. Note that we must ensure we always work with
	// double precision scale vectors.
	//
	// TODO:  how can we check if the precision of Vector is already
	//        double, so that we can save extra copies.
	if (doMC64) {
		CPUTimer loc_timer;
		loc_timer.Start();
		DoubleVector  mc64RowScaleD;
		DoubleVector  mc64ColScaleD;
		MC64(Acoo, scale, mc64FirstStageOnly, mc64RowPerm, mc64RowScaleD, mc64ColScaleD, scaleMap);
		mc64RowScale = mc64RowScaleD;
		mc64ColScale = mc64ColScaleD;
		loc_timer.Stop();
		m_timeMC64 = loc_timer.getElapsed();
	} else {
		mc64RowScale.resize(m_n);
		mc64ColScale.resize(m_n);
		mc64RowPerm.resize(m_n);
		scaleMap.resize(m_nnz);

		thrust::sequence(mc64RowPerm.begin(), mc64RowPerm.end());
		cusp::blas::fill(mc64RowScale, (T) 1.0);
		cusp::blas::fill(mc64ColScale, (T) 1.0);
		cusp::blas::fill(scaleMap, (T) 1.0);
	}

	k_mc64 = 0;
	for (EdgeIterator edgeIt = m_edges.begin(); edgeIt != m_edges.end(); edgeIt++)
		if (k_mc64 < abs(edgeIt->m_from - edgeIt->m_to))
			k_mc64 = abs(edgeIt->m_from - edgeIt->m_to);

	if (testMC64)
		return k_mc64;

	// Apply reverse Cuthill-McKee algorithm.
	int bandwidth = RCM(m_edges, optReordering, optPerm);

	// Initialize the iterator m_first (in case dropOff() is not called).
	m_first = m_edges.begin();

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
// The return value is the number of dropped diagonals. We also return the
// actual norm reduction fraction achieved.
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
	std::sort(m_edges.begin(), m_edges.end(), CompareEdgeLength());

	// Calculate the 1-norm of the current matrix and the minimum norm that
	// must be retained after drop-off. Initialize the 1-norm of the resulting
	// truncated matrix.
	T norm_in = std::accumulate(m_edges.begin(), m_edges.end(), (T) 0, AccumulateEdgeValue());
	T min_norm_out = (1 - frac) * norm_in;
	T norm_out = norm_in;

	// Walk all edges and accumulate the weigth (1-norm) of one band at a time.
	// Continue until we are left with the main diagonal only or until the weight
	// of all proccessed bands exceeds the allowable drop off (provided we do not
	// exceed the specified maximum bandwidth.
	m_first = m_edges.begin();
	EdgeIterator  last = m_first;
	int           dropped = 0;

	while (true) {
		// Current band
		int bandwidth = abs(m_first->m_from - m_first->m_to);

		// Stop now if we reached the main diagonal.
		if (bandwidth == 0)
			break;

		// Find all edges in the current band and calculate the norm of the band.
		do {last++;} while(abs(last->m_from - last->m_to) == bandwidth);

		T band_norm = std::accumulate(m_first, last, (T) 0, AccumulateEdgeValue());

		// Stop now if we are below the specified maximum and removing this band
		// would reduce the norm by more than allowed.
		if (bandwidth <= maxBandwidth && norm_out - band_norm < min_norm_out)
			break;

		// Remove the norm of this band and move to the next one.
		norm_out -= band_norm;
		m_first = last;
		dropped++;
	}

	timer.Stop();
	m_timeDropoff = timer.getElapsed();

	// Calculate the actual norm reduction fraction.
	frac_actual = 1 - norm_out/norm_in;

	return dropped;
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

	EdgeIterator first = m_first;

	int partSize = m_n / numPartitions;
	int remainder = m_n % numPartitions;

	m_major_edges.clear();

	if (m_trackReordering) {
		typeMap.resize(m_nnz);
		offDiagMap.resize(m_nnz);
		WVMap.resize(m_nnz);
	}

	m_exists.resize(m_n);
	cusp::blas::fill(m_exists, false);

	for (EdgeIterator it = first; it != m_edges.end(); ++it) {
		int j = it->m_from;
		int l = it->m_to;
		int curPartNum = l / (partSize + 1);
		if (curPartNum >= remainder)
			curPartNum = remainder + (l-remainder * (partSize + 1)) / partSize;

		int curPartNum2 = j / (partSize + 1);
		if (curPartNum2 >= remainder)
			curPartNum2 = remainder + (j-remainder * (partSize + 1)) / partSize;

		if (curPartNum == curPartNum2)
			m_major_edges.push_back(*it);
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
					typeMap[it->m_ori_idx] = 0;
					offDiagMap[it->m_ori_idx] = curPartNum2*2*bandwidth*bandwidth + (j+bandwidth-partEndRow) * bandwidth + (l-partStartCol);
					WVMap[it->m_ori_idx] = curPartNum2*2*bandwidth*bandwidth + (j+bandwidth-partEndRow) + offDiagReorderings_right[curPartNum2*bandwidth+l-partStartCol] * bandwidth;
				}

				offDiags_host[curPartNum2*2*bandwidth*bandwidth + (j+bandwidth-partEndRow) * bandwidth + (l-partStartCol)] = WV_host[curPartNum2*2*bandwidth*bandwidth + (j+bandwidth-partEndRow) + offDiagReorderings_right[curPartNum2*bandwidth+l-partStartCol] * bandwidth] = it->m_val;

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
					typeMap[it->m_ori_idx] = 0;
					offDiagMap[it->m_ori_idx] = (curPartNum*2+1)*bandwidth*bandwidth + (j-partStartRow) * bandwidth + (l-partEndCol+bandwidth);
					WVMap[it->m_ori_idx] = (curPartNum*2+1)*bandwidth*bandwidth + (j-partStartRow) + (offDiagReorderings_left[(curPartNum2-1)*bandwidth+l-partEndCol+bandwidth]) * bandwidth;
				}

				offDiags_host[(curPartNum*2+1)*bandwidth*bandwidth + (j-partStartRow) * bandwidth + (l-partEndCol+bandwidth)] = WV_host[(curPartNum*2+1)*bandwidth*bandwidth + (j-partStartRow) + (offDiagReorderings_left[(curPartNum2-1)*bandwidth+l-partEndCol+bandwidth]) * bandwidth] = it->m_val;
			}
		}
	}
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
	{
		secondPerm.resize(m_n);
		cusp::blas::fill(secondPerm, 0);
		for (EdgeIterator edgeIt = m_major_edges.begin(); edgeIt != m_major_edges.end(); edgeIt++)
			secondPerm[edgeIt -> m_to]++;
		thrust::inclusive_scan(secondPerm.begin(), secondPerm.end(), secondPerm.begin());
		EdgeVector tmp_edges(m_major_edges.size());
		for (EdgeRevIterator edgeIt = m_major_edges.rbegin(); edgeIt != m_major_edges.rend(); edgeIt++)
			tmp_edges[--secondPerm[edgeIt->m_to]] = *edgeIt;
		m_major_edges = tmp_edges;
	}

	int node_begin = 0, node_end;
	int partSize = m_n / numPartitions;
	int remainder = m_n % numPartitions;
	EdgeIterator edgeBegin = m_major_edges.begin(), edgeEnd;
	secondReorder.resize(m_n);

	for (int i = 0; i < numPartitions; i++) {
		if (i < remainder)
			node_end = node_begin + partSize + 1;
		else
			node_end = node_begin + partSize;

		for (edgeEnd = edgeBegin; edgeEnd != m_major_edges.end(); edgeEnd++) {
			if (edgeEnd->m_from >= node_end)
				break;
		}

		partitionedRCM(edgeBegin,
		               edgeEnd,
		               node_begin,
		               node_end,
		               secondReorder,
		               secondPerm);

		node_begin = node_end;
		edgeBegin = edgeEnd;
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

	size_t idx = 0;
	Acoo.resize(m_n, m_n, m_edges.end() - m_first);
	Acoo.num_rows = Acoo.num_cols = m_n;
	Acoo.num_entries = m_edges.end() - m_first;

	if (m_trackReordering) {
		for (EdgeIterator it = m_first; it != m_edges.end(); ++it, idx++) {
			int j = it->m_from;
			int l = it->m_to;

			size_t i = (size_t) l * (2 * bandwidth + 1) + bandwidth + j - l;
			typeMap[it->m_ori_idx] = 1;
			bandedMatMap[it->m_ori_idx] = i;

			Acoo.row_indices[idx] = it->m_from;
			Acoo.column_indices[idx] = it->m_to;
			Acoo.values[idx] = it->m_val;

			if (ks_col[l] < j - l)
				ks_col[l] = j - l;
			if (ks_row[j] < l-j)
				ks_row[j] = l-j;
		}
	} else {
		for (EdgeIterator it = m_first; it != m_edges.end(); ++it, ++idx) {
			int j = it->m_from;
			int l = it->m_to;

			Acoo.row_indices[idx] = it->m_from;
			Acoo.column_indices[idx] = it->m_to;
			Acoo.values[idx] = it->m_val;

			if (ks_col[l] < j - l)
				ks_col[l] = j - l;
			if (ks_row[j] < l-j)
				ks_row[j] = l-j;
		}
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

	EdgeIterator toStart = m_major_edges.begin(), toEnd = m_major_edges.end();

	Acoo.resize(m_n, m_n, toEnd - toStart);
	Acoo.num_rows = Acoo.num_cols = m_n;
	Acoo.num_entries = toEnd - toStart;

	for (EdgeIterator it = toStart; it != toEnd; ++it) {
		int j = it->m_from;
		int l = it->m_to;
		int curPartNum = l / (partSize + 1);
		if (curPartNum >= remainder)
			curPartNum = remainder + (l-remainder * (partSize + 1)) / partSize;
		if (ks[curPartNum] < abs(l-j))
			ks[curPartNum] = abs(l-j);
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

	size_t idx = 0;
	for (EdgeIterator it = toStart; it != toEnd; ++it, ++idx) {
		int j = it->m_from;
		int l = it->m_to;

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

		Acoo.row_indices[idx] = it->m_from;
		Acoo.column_indices[idx] = it->m_to;
		Acoo.values[idx] = it->m_val;

		if (ks_col[l] < j - l)
			ks_col[l] = j - l;
		if (ks_row[j] < l-j)
			ks_row[j] = l-j;

		if (m_trackReordering) {
			typeMap[it->m_ori_idx] = 1;
			bandedMatMap[it->m_ori_idx] = i;
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
Graph<T>::MC64(const MatrixCoo& Acoo,
			   bool             scale,
			   bool             mc64FirstStageOnly,
               IntVector&       mc64RowPerm,
               DoubleVector&    mc64RowScale,
               DoubleVector&    mc64ColScale,
               MatrixMapF&      scaleMap)
{
	find_minimum_match(Acoo, mc64FirstStageOnly, mc64RowPerm, mc64RowScale, mc64ColScale);

	CPUTimer loc_timer;

	loc_timer.Start();

	if (m_trackReordering)
		scaleMap.resize(m_nnz);

	// TODO: how to do scale when we apply only the first stage
	if (mc64FirstStageOnly)
		scale = false;

	if (scale) {
		for (EdgeIterator iter = m_edges.begin(); iter != m_edges.end(); iter++) {
			int from = iter->m_from;
			int to   = iter->m_to;
			T scaleFact = (T)(mc64RowScale[from] * mc64ColScale[to]);
			if (m_trackReordering)
				scaleMap[iter->m_ori_idx] = scaleFact;
			iter->m_val *= scaleFact;
			iter->m_from = mc64RowPerm[from];
		}
	} else {
		for(EdgeIterator iter = m_edges.begin(); iter != m_edges.end(); iter++)
			iter->m_from = mc64RowPerm[iter->m_from];

		if (m_trackReordering)
			cusp::blas::fill(scaleMap, (T) 1.0);
	}
	loc_timer.Stop();
	m_timeMC64_post = loc_timer.getElapsed();

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
Graph<T>::RCM(EdgeVector&  edges,
              IntVector&   optReordering,
              IntVector&   optPerm)
{
	optReordering.resize(m_n);
	optPerm.resize(m_n);

	int nnz = edges.size();

	IntVector tmp_reordering(m_n);
	IntVector degrees(m_n, 0);

	thrust::sequence(optReordering.begin(), optReordering.end());

	std::vector<int> *in_out_graph;

	in_out_graph = new std::vector<int> [m_n];

	EdgeIterator begin = edges.begin();
	EdgeIterator end = edges.end();
	int tmp_bdwidth = thrust::transform_reduce(begin, end, EdgeLength(), 0, thrust::maximum<int>());
	int bandwidth = tmp_bdwidth;
	buildTopology(begin, end, degrees, in_out_graph);

	const int MAX_NUM_TRIAL = 10;
	const int BANDWIDTH_THRESHOLD = 256;
	const int BANDWIDTH_MIN_REQUIRED = 10000;

	CPUTimer timer;
	timer.Start();

	BoolVector tried(m_n, false);
	tried[0] = true;

	int last_tried = 0;

	for (int trial_num = 0; trial_num < MAX_NUM_TRIAL || (bandwidth >= BANDWIDTH_MIN_REQUIRED && trial_num < 10*MAX_NUM_TRIAL); trial_num++)
	{
		std::queue<int> q;
		std::priority_queue<NodeType> pq;

		int tmp_node;
		BoolVector pushed(m_n, false);

		int left_cnt = m_n;
		int j = 0, last = 0;

		if (trial_num > 0) {

			if (trial_num < MAX_NUM_TRIAL) {
				tmp_node = rand() % m_n;

				while(tried[tmp_node])
					tmp_node = rand() % m_n;
			} else {
				if (last_tried >= m_n - 1) {
					fprintf(stderr, "All possible starting points have been tried in RCM\n");
					break;
				}
				for (tmp_node = last_tried+1; tmp_node < m_n; tmp_node++)
					if (!tried[tmp_node]) {
						last_tried = tmp_node;
						break;
					}
			}

			pushed[tmp_node] = true;
			tried[tmp_node] = true;
			q.push(tmp_node);
		}

		while(left_cnt--) {
			if(q.empty()) {
				left_cnt++;
				int i;

				for(i = last; i < m_n; i++) {
					if(!pushed[i]) {
						q.push(i);
						pushed[i] = true;
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

			std::vector<int> &tmp_vec = in_out_graph[tmp_node];
			int out_size = tmp_vec.size();
			if(out_size != 0) {
				for (int i = 0; i < out_size; i++)  {
					int target_node = tmp_vec[i];
					if(!pushed[target_node]) {
						pushed[target_node] = true;
						pq.push(NodeType(target_node, degrees[target_node]));
					}
				}
			}

			while(!pq.empty()) {
				q.push(pq.top().m_idx);
				pq.pop();
			}
		}

		thrust::scatter(thrust::make_counting_iterator(0), 
						thrust::make_counting_iterator(m_n),
						tmp_reordering.begin(),
						optPerm.begin());

		{
			int *perm_array = thrust::raw_pointer_cast(&optPerm[0]);
			tmp_bdwidth = thrust::transform_reduce(edges.begin(), edges.end(), PermutedEdgeLength(perm_array), 0, thrust::maximum<int>());
		}

		if(bandwidth > tmp_bdwidth) {
			bandwidth = tmp_bdwidth;
			optReordering = tmp_reordering;
		}

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
		thrust::for_each(edges.begin(), edges.end(), PermuteEdge(perm_array));
	}

	delete [] in_out_graph;

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
Graph<T>::partitionedRCM(EdgeIterator&  begin,
                         EdgeIterator&  end,
                         int            node_begin,
                         int            node_end,
                         IntVector&     optReordering,
                         IntVector&     optPerm)
{
	static std::vector<int> tmp_reordering;
	tmp_reordering.resize(m_n);

	static IntVector degrees;
	if (degrees.size() != m_n)
		degrees.resize(m_n, 0);
	else
		cusp::blas::fill(degrees, 0);

	// for(int i = node_begin; i < node_end; i++)
		// optReordering[i] = i;
	thrust::sequence(optReordering.begin()+node_begin, optReordering.begin()+node_end, node_begin);

	std::vector<int>* in_out_graph;

	in_out_graph = new std::vector<int> [node_end];

	int tmp_bdwidth = thrust::transform_reduce(begin, end, EdgeLength(), 0, thrust::maximum<int>());
	int opt_bdwidth = tmp_bdwidth;
	buildTopology(begin, end, degrees, in_out_graph);

	const int MAX_NUM_TRIAL = 10;
	const int BANDWIDTH_THRESHOLD = 128;

	static BoolVector tried(m_n, false);
	if (tried.size() != m_n)
		tried.resize(m_n, false);
	else
		cusp::blas::fill(tried, false);

	tried[node_begin] = true;

	CPUTimer timer;
	timer.Start();

	for (int num_trial = 0; num_trial < MAX_NUM_TRIAL; num_trial++) {
		std::queue<int> q;
		std::priority_queue<NodeType> pq;

		int tmp_node;
		BoolVector pushed(node_end, false);

		if (num_trial > 0) {
			tmp_node = rand() % (node_end - node_begin) + node_begin;
			while (tried[tmp_node])
				tmp_node = rand() % (node_end - node_begin) + node_begin;
			tried[tmp_node] = pushed[tmp_node] = true;
			q.push(tmp_node);
		}

		int left_cnt = node_end - node_begin;
		int j = node_begin, last = node_begin;

		while(left_cnt--) {
			if(q.empty()) {
				left_cnt++;
				int i;
				for(i = last; i < node_end; i++) {
					if(!pushed[i]) {
						q.push(i);
						pushed[i] = true;
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

			std::vector<int>& tmp_vec = in_out_graph[tmp_node];
			int in_out_size = tmp_vec.size();
			if(in_out_size != 0) {
				for (int i = 0; i < in_out_size; i++)  {
					int target_node = tmp_vec[i];
					if(!pushed[target_node]) {
						pushed[target_node] = true;
						pq.push(NodeType(target_node, degrees[target_node]));
					}
				}
			}

			while(!pq.empty()) {
				q.push(pq.top().m_idx);
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

		if(opt_bdwidth <= BANDWIDTH_THRESHOLD)
			break;
	}

	timer.Stop();
	m_timeRCM += timer.getElapsed();

	thrust::scatter(thrust::make_counting_iterator(node_begin),
	                thrust::make_counting_iterator(node_end),
	                optReordering.begin() + node_begin,
	                optPerm.begin());

	{
		int* perm_array = thrust::raw_pointer_cast(&optPerm[0]);
		thrust::for_each(begin, end, PermuteEdge(perm_array));
	}

	delete [] in_out_graph;

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
                        IntVector&         degrees,
                        std::vector<int>*  in_out_graph)
{
	for(EdgeIterator edgeIt = begin; edgeIt != end; edgeIt++) {
		int from = edgeIt -> m_from, to = edgeIt -> m_to;
		if (from != to) {
			in_out_graph[from].push_back(to);
			in_out_graph[to].push_back(from);
			degrees[from]++;
			degrees[to]++;
		}
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
Graph<T>::find_minimum_match(const MatrixCoo& Acoo,
							 bool             mc64FirstStageOnly, 
							 IntVector&       mc64RowPerm,
                             DoubleVector&    mc64RowScale,
                             DoubleVector&    mc64ColScale)
{
	CPUTimer loc_timer;

	loc_timer.Start();
	// Allocate space for the output vectors.
	mc64RowPerm.resize(m_n, 0);
	mc64RowScale.resize(m_n + 1, 0);
	mc64ColScale.resize(m_n + 1, 0);

	// Allocate space for temporary vectors.
	IntVector     row_ptr(m_n + 1, 0);
	IntVector     rows(m_nnz);
	IntVector     rev_match_nodes(m_nnz);
	DoubleVector  c_val(m_nnz);
	DoubleVector  max_val_in_col(m_n + 1, 0);
	IntVector     prev(m_n + 1);
	IntVector     matched(m_n + 1, 0);
	IntVector     rev_matched(m_n + 1, 0);

	get_csc_matrix(Acoo, row_ptr, rows, c_val, max_val_in_col);
	loc_timer.Stop();
	m_timeMC64_pre = loc_timer.getElapsed();

	loc_timer.Start();
	init_reduced_cval(mc64FirstStageOnly, row_ptr, rows, c_val, mc64RowScale, mc64ColScale, mc64RowPerm, rev_match_nodes, matched, rev_matched);
	loc_timer.Stop();
	m_timeMC64_first = loc_timer.getElapsed();

	loc_timer.Start();
	if (!mc64FirstStageOnly) {
		IntVector  irn(m_n);
		for(int i=0; i<m_n; i++) {
			if(rev_matched[i]) continue;
			bool success = find_shortest_aug_path(i, matched, rev_matched, mc64RowPerm, rev_match_nodes, row_ptr, rows, prev, mc64RowScale, mc64ColScale, c_val, irn);
		}
	}

	{
		for (int i=0; i<m_n; i++)
			if (!matched[i])
				throw system_error(system_error::Matrix_singular, "Singular matrix found");
	}

	mc64RowScale.pop_back();
	mc64ColScale.pop_back();
	max_val_in_col.pop_back();

	thrust::transform(mc64RowScale.begin(), mc64RowScale.end(), mc64RowScale.begin(), Exponential());
	thrust::transform(thrust::make_transform_iterator(mc64ColScale.begin(), Exponential()),
	                  thrust::make_transform_iterator(mc64ColScale.end(), Exponential()),
	                  max_val_in_col.begin(),
	                  mc64ColScale.begin(),
	                  thrust::divides<double>());
	loc_timer.Stop();
	m_timeMC64_second = loc_timer.getElapsed();
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
Graph<T>::get_csc_matrix(const MatrixCoo& Acoo,
						 IntVector&       row_ptr,
                         IntVector&       rows,
                         DoubleVector&    c_val,
                         DoubleVector&    max_val_in_col)
{
	int nnz = Acoo.num_entries;

	cusp::blas::fill(c_val, LOC_INFINITY);

	for (int i = 0; i < nnz; i++)
		row_ptr[Acoo.column_indices[i]]++;

	const int GPU_ASSEMBLE_THRESHOLD = 100000;
	if (nnz < GPU_ASSEMBLE_THRESHOLD) {
		thrust::exclusive_scan(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
		for (int i = 0; i < nnz; i++) {
			int from = Acoo.row_indices[i];
			int to = Acoo.column_indices[i];
			double tmp_val = fabs(Acoo.values[i]);
			rows[row_ptr[to]++] = from;

			if (max_val_in_col[to] < tmp_val)
				max_val_in_col[to] = tmp_val;
		}

		for (int i = nnz-1; i >= 0; i--) {
			int to = Acoo.column_indices[i];
			c_val[--row_ptr[to]] = log(max_val_in_col[to] / fabs(Acoo.values[i]));
		}
	} else {
		thrust::inclusive_scan(row_ptr.begin(), row_ptr.end(), row_ptr.begin());

		for (int i = nnz - 1; i >= 0; i--) {
			int from = Acoo.row_indices[i];
			int to = Acoo.column_indices[i];
			double tmp_val = fabs(Acoo.values[i]);
			rows[row_ptr[to]-1] = from;
			c_val[--row_ptr[to]] = tmp_val;

			if (max_val_in_col[to] < tmp_val)
				max_val_in_col[to] = tmp_val;
		}

		DoubleVectorD dc_val     = c_val;
		DoubleVectorD d_max_vals = max_val_in_col;
		IntVectorD    d_row_ptr  = row_ptr;

		double *dc_val_ptr = thrust::raw_pointer_cast(&dc_val[0]);
		double *dmax_val_ptr = thrust::raw_pointer_cast(&d_max_vals[0]);
		int *d_row_ptrs = thrust::raw_pointer_cast(&d_row_ptr[0]);

		int blockX = m_n, blockY = 1;
		kernelConfigAdjust(blockX, blockY, 32768);
		dim3 grids(blockX, blockY);

		device::getResidualValues<<<grids, 64>>>(m_n, dc_val_ptr, dmax_val_ptr, d_row_ptrs);

		c_val = dc_val;
	}
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
							IntVector&     row_ptr,
                            IntVector&     rows,
                            DoubleVector&  c_val,
                            DoubleVector&  u_val,
                            DoubleVector&  v_val,
                            IntVector&     match_nodes,
                            IntVector&     rev_match_nodes,
                            IntVector&     matched,
                            IntVector&     rev_matched) 
{
	cusp::blas::fill(u_val, LOC_INFINITY);
	cusp::blas::fill(v_val, LOC_INFINITY);

	for(int i = 0; i < m_n; i++) {
		int start_idx = row_ptr[i], end_idx = row_ptr[i+1];
		for(int j = start_idx; j < end_idx; j++) {
			if (c_val[j] > LOC_INFINITY / 2.0) continue;
			int row = rows[j];
			if(u_val[row] > c_val[j]) {
				u_val[row] = c_val[j];
			}
		}
	}

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
		for(int i = 0; i < m_n; i++) {
			if (!matched[i])
				u_val[i] = 0.0;
			if (!rev_matched[i])
				v_val[i] = 0.0;
		}
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
                                 IntVector&     matched,
                                 IntVector&     rev_matched,
                                 IntVector&     match_nodes,
                                 IntVector&     rev_match_nodes,
                                 IntVector&     row_ptr,
                                 IntVector&     rows,
                                 IntVector&     prev,
                                 DoubleVector&  u_val,
                                 DoubleVector&  v_val,
                                 DoubleVector&  c_val,
                                 IntVector&     irn)
{
	bool success = false;

	static IntVector B(m_n+1, 0);
	int b_cnt = 0;
	static BoolVector inB(m_n+1, false);

	std::priority_queue<Dijkstra> Q;
	double lsp = 0.0;
	double lsap = LOC_INFINITY;
	int cur_node = init_node;

	int i;

	int isap = -1;
	int ksap = -1;
	prev[init_node] = -1;

	static DoubleVector d_vals(m_n+1, LOC_INFINITY);
	static BoolVector visited(m_n+1, false);

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
					Q.push(Dijkstra(cur_row, d_new));
					irn[cur_row] = i;
				}
			}
		}

		Dijkstra min_d;
		bool found = false;

		while(!Q.empty()) {
			min_d = Q.top();
			Q.pop();
			if(visited[min_d.m_idx]) 
				continue;
			found = true;
			break;
		}
		if(!found)
			break;

		int tmp_idx = min_d.m_idx;
		visited[tmp_idx] = true;

		lsp = min_d.m_val;
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
			d_vals[tmpD.m_idx] = LOC_INFINITY;
		}
	}

	return success;
}


} // namespace spike


#endif
