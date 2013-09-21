#ifndef SPIKE_COMPONENTS_H
#define SPIKE_COMPONENTS_H

#include <map>

#include <cusp/array1d.h>
#include <thrust/sequence.h>


namespace spike {


// ----------------------------------------------------------------------------
// Components
//
// This structure encapsulate information about matrix component, representing
// decoupled diagonal blocks which can be processed independently.
// ----------------------------------------------------------------------------
struct Components
{
	typedef typename cusp::array1d<int, cusp::host_memory> VectorI;

	Components(int n) : m_n(n) {
		m_compIndices.resize(m_n);
		thrust::sequence(m_compIndices.begin(), m_compIndices.end());
		m_numComponents = m_n;
	}

	int getComponentIndex(int node) {
		if (m_compIndices[node] == node)
			return node;
		m_compIndices[node] = getComponentIndex(m_compIndices[node]);
		return m_compIndices[node];
	}

	void combineComponents(int node1, int node2) {
		int r1 = getComponentIndex(node1), r2 = getComponentIndex(node2);

		if (r1 != r2) {
			m_compIndices[r1] = r2;
			m_numComponents --;
		}
	}

	void adjustComponentIndices() {
		for (int i = 0; i < m_n; i++)
			m_compIndices[i] = getComponentIndex(i);
	
		std::map<int, int> compIndicesMapping;
		VectorI            compCounts(m_numComponents, 0);
	
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

	VectorI     m_compIndices;
	int         m_n;
	int         m_numComponents;
};



} // namespace spike


#endif
