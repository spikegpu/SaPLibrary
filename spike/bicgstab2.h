#ifndef SPIKE_BICGSTAB_2_H
#define SPIKE_BICGSTAB_2_H

#include <vector>

#include <cusp/blas.h>
#include <cusp/print.h>
#include <cusp/array1d.h>

#include <spike/monitor.h>
#include <spike/precond.h>


namespace spike {

template <typename T>
struct IsEqual
{
	T m_val;
	IsEqual(T val = 0):m_val(val) {}

	__host__ __device__
	bool operator() (const T &val)
	{
		return m_val == val;
	}
};

template <typename Vector>
void precondSolveWrapper(Vector& rhs,
						 Vector& sol,
						 std::vector<Precond<Vector> *>&					precond_pointers,
						 cusp::array1d<int, typename Vector::memory_space>&	compIndices,
						 cusp::array1d<int, typename Vector::memory_space>&	comp_perms,
						 std::vector<cusp::array1d<int, typename Vector::memory_space> >&	comp_reorderings)
{
	typedef typename Vector::value_type   ValueType;

	int numComponents = comp_reorderings.size();

	for (int i=0; i<numComponents; i++) {
		int loc_n = comp_reorderings[i].size();

		Vector buffer_rhs(loc_n);
		Vector buffer_sol(loc_n);

		thrust::scatter_if(rhs.begin(), rhs.end(), comp_perms.begin(), compIndices.begin(), buffer_rhs.begin(), IsEqual<int>(i));
		precond_pointers[i]->solve(buffer_rhs, buffer_sol);
		thrust::scatter(buffer_sol.begin(), buffer_sol.end(), comp_reorderings[i].begin(), sol.begin());
	}
}


// ----------------------------------------------------------------------------
// bicgstabl()
//
// This function implements a preconditioned BiCGStab(l) Krylov method.
// ----------------------------------------------------------------------------
template <typename SpmvOperator, typename Vector, int L>
void bicgstabl(SpmvOperator&     spmv,
               const Vector&     b,
               Vector&           x,
               Monitor<Vector>&  monitor,
			   std::vector<Precond<Vector> *>&		    precond_pointers,
			   cusp::array1d<int, cusp::host_memory>&	compIndices,
			   cusp::array1d<int, cusp::host_memory>&	comp_perms,
			   std::vector<cusp::array1d<int, cusp::host_memory> >&	comp_reorderings)
{
	using namespace cusp;

	typedef typename Vector::value_type   ValueType;
	typedef typename Vector::memory_space MemorySpace;


	// Allocate workspace
	int   n = b.size();

	ValueType rou0 = ValueType(1);
	ValueType alpha = ValueType(0);
	ValueType omega = ValueType(1);
	ValueType rou1;

	Vector r0(n);
	Vector r(n);
	Vector u(n,0);
	Vector xx(n);
	Vector Pv(n);

	array1d<int, MemorySpace> loc_compIndices = compIndices;
	array1d<int, MemorySpace> loc_comp_perms = comp_perms;
	std::vector<array1d<int, MemorySpace> > loc_comp_reorderings;

	int numComponents = comp_reorderings.size();
	for (int i=0; i < numComponents; i++)
		loc_comp_reorderings.push_back(comp_reorderings[i]);

	std::vector<Vector> rr(L+1), uu(L+1);
	for(int k = 0; k <= L; k++) {
		rr[k].resize(n, 0);
		uu[k].resize(n, 0);
	}

	ValueType tao[L+1][L+1];
	ValueType gamma[L+2];
	ValueType gamma_prime[L+2];
	ValueType gamma_primeprime[L+2];
	ValueType sigma[L+2];

	// r0 <- b - A * x
	spmv(x, r0);
	blas::axpby(b, r0, r0, ValueType(1), ValueType(-1));

	// r <- r0
	blas::copy(r0, r);

	// uu(0) <- u
	// rr(0) <- r
	// xx <- x
	thrust::copy(
			thrust::make_zip_iterator(thrust::make_tuple(u.begin(), x.begin(), r.begin())), 
			thrust::make_zip_iterator(thrust::make_tuple(u.end(), x.end(), r.end())), 
			thrust::make_zip_iterator(thrust::make_tuple(uu[0].begin(), xx.begin(), rr[0].begin()))
			);

	while(!monitor.done(r)) {

		rou0 = -omega * rou0;

		monitor.increment(0.25f);

		for(int j = 0; j < L; j++) {
			rou1 = blas::dotc(rr[j], r0);

			// return with failure
			if(rou0 == 0)
				return;

			ValueType beta = alpha * rou1 / rou0;
			rou0 = rou1;

			for(int i = 0; i <= j; i++) {
				// uu(i) = rr(i) - beta * uu(i)
				blas::axpby(rr[i], uu[i], uu[i], ValueType(1), -beta);
			}

			// uu(j+1) <- A * P^(-1) * uu(j);
			// precond.solve(uu[j], Pv);
			precondSolveWrapper(uu[j], Pv, precond_pointers, loc_compIndices, loc_comp_perms, loc_comp_reorderings);
			spmv(Pv, uu[j+1]);

			// gamma <- uu(j+1) . r0;
			ValueType gamma = blas::dotc(uu[j+1], r0);
			if(gamma == 0)
				return;

			alpha = rou0 / gamma;

			for(int i = 0; i <= j; i++) {
				// rr(i) <- rr(i) - alpha * uu(i+1)
				blas::axpy(uu[i+1], rr[i], ValueType(-alpha));
			}

			// rr(j+1) = A * P^(-1) * rr(j)
			//precond.solve(rr[j], Pv);
			precondSolveWrapper(rr[j], Pv, precond_pointers, loc_compIndices, loc_comp_perms, loc_comp_reorderings);
			spmv(Pv, rr[j+1]);
			
			// xx <- xx + alpha * uu(0)
			blas::axpy(uu[0], xx, alpha);

			if(monitor.done(rr[0])) {
				//precond.solve(xx, x);
				precondSolveWrapper(xx, x, precond_pointers, loc_compIndices, loc_comp_perms, loc_comp_reorderings);
				return;
			}
		}


		for(int j = 1; j <= L; j++) {
			for(int i = 1; i < j; i++) {
				tao[i][j] = blas::dotc(rr[j], rr[i]) / sigma[i];
				blas::axpy(rr[i], rr[j], -tao[i][j]);
			}
			sigma[j] = blas::dotc(rr[j], rr[j]);
			if(sigma[j] == 0)
				return;
			gamma_prime[j] = blas::dotc(rr[j], rr[0]) / sigma[j];
		}

		gamma[L] = gamma_prime[L];
		omega = gamma[L];

		for(int j = L-1; j > 0; j--) {
			gamma[j] = gamma_prime[j];
			for(int i = j+1; i <= L; i++)
				gamma[j] -= tao[j][i] * gamma[i];
		}

		for(int j = 1; j < L; j++) {
			gamma_primeprime[j] = gamma[j+1];
			for(int i = j+1; i < L; i++)
				gamma_primeprime[j] += tao[j][i] * gamma[i+1];
		}

		// xx    <- xx    + gamma * rr(0)
		// rr(0) <- rr(0) - gamma'(L) * rr(L)
		// uu(0) <- uu(0) - gamma(L) * uu(L)
		blas::axpy(rr[0], xx,    gamma[1]);
		blas::axpy(rr[L], rr[0], -gamma_prime[L]);
		blas::axpy(uu[L], uu[0], -gamma[L]);

		monitor.increment(0.25f);

		if (monitor.done(rr[0])) {
			// precond.solve(xx, x);
			precondSolveWrapper(xx, x, precond_pointers, loc_compIndices, loc_comp_perms, loc_comp_reorderings);
			return;
		}

		monitor.increment(0.25f);

		// uu(0) <- uu(0) - sum_j { gamma(j) * uu(j) }
		// xx    <- xx    + sum_j { gamma''(j) * rr(j) }
		// rr(0) <- rr(0) - sum_j { gamma'(j) * rr(j) }
		for(int j = 1; j < L; j++) {
			blas::axpy(uu[j], uu[0],  -gamma[j]);
			blas::axpy(rr[j], xx,     gamma_primeprime[j]);
			blas::axpy(rr[j], rr[0],  -gamma_prime[j]);

			if (monitor.done(rr[0])) {
				// precond.solve(xx, x);
				precondSolveWrapper(xx, x, precond_pointers, loc_compIndices, loc_comp_perms, loc_comp_reorderings);
				return;
			}
		}

		// u <- uu(0)
		// x <- xx
		// r <- rr(0)
		thrust::copy(
				thrust::make_zip_iterator(thrust::make_tuple(uu[0].begin(), xx.begin(), rr[0].begin())), 
				thrust::make_zip_iterator(thrust::make_tuple(uu[0].end(), xx.end(), rr[0].end())), 
				thrust::make_zip_iterator(thrust::make_tuple(u.begin(), x.begin(), r.begin()))
				);

		monitor.increment(0.25f);
	}
}


// ----------------------------------------------------------------------------
// Specializations of the generic BiCGStab(L) function.
// ----------------------------------------------------------------------------
template <typename SpmvOperator, typename Vector>
void bicgstab2(SpmvOperator&     spmv,
               const Vector&     b,
               Vector&           x,
               Monitor<Vector>&  monitor,
			   std::vector<Precond<Vector>*>&		    precond_pointers,
			   cusp::array1d<int, cusp::host_memory>&	compIndices,
			   cusp::array1d<int, cusp::host_memory>&	comp_perms,
			   std::vector<cusp::array1d<int, cusp::host_memory> >&	comp_reorderings
			   )
{
	bicgstabl<SpmvOperator, Vector, 2>(spmv, b, x, monitor, precond_pointers, compIndices, comp_perms, comp_reorderings);
}

template <typename SpmvOperator, typename Vector>
void bicgstab4(SpmvOperator&     spmv,
               const Vector&     b,
               Vector&           x,
               Monitor<Vector>&  monitor,
			   std::vector<Precond<Vector>*>&		    precond_pointers,
			   cusp::array1d<int, cusp::host_memory>&	compIndices,
			   cusp::array1d<int, cusp::host_memory>&	comp_perms,
			   std::vector<cusp::array1d<int, cusp::host_memory> >&	comp_reorderings
			   )
{
	bicgstabl<SpmvOperator, Vector, 4>(spmv, b, x, monitor, precond_pointers, compIndices, comp_perms, comp_reorderings);
}



} // namespace spike



#endif

