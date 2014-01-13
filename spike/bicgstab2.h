/** \file bicgstab2.h
 *  \brief BiCGStab(L) preconditioned iterative Krylov solver.
 */

#ifndef SPIKE_BICGSTAB_2_H
#define SPIKE_BICGSTAB_2_H

#include <vector>

#include <cusp/blas.h>
#include <cusp/print.h>
#include <cusp/array1d.h>

#include <spike/monitor.h>
#include <spike/precond.h>


namespace spike {

typedef typename cusp::array1d<int,  cusp::host_memory>  IntVectorH;


template <typename T>
struct IsEqualTo
{
	T m_val;

	IsEqualTo(T val = 0) : m_val(val) {}

	__host__ __device__
	bool operator() (const T& val)
	{
		return m_val == val;
	}
};


/// Preconditioned BiCGStab(L) Krylov method
/**
 * \tparam SpmvOperator is a functor class for sparse matrix-vector product.
 * \tparam SolverVector is the vector type for the linear system solution.
 * \tparam PrecVector is the vector type used in the preconditioner.
 * \tparam L is the degree of the BiCGStab(L) method.
 */
template <typename SpmvOperator, typename SolverVector, typename PrecVector, int L>
void bicgstabl(SpmvOperator&                       spmv,
               SolverVector&                       x,
               const SolverVector&                 b,
               Monitor<SolverVector>&              monitor,
               Precond<PrecVector>&                precond)
{
	typedef typename SolverVector::value_type   SolverValueType;
	typedef typename SolverVector::memory_space MemorySpace;

	typedef typename cusp::array1d<int, MemorySpace>  IntVector;

	// Allocate workspace
	int  n = b.size();

	SolverValueType rho0  = SolverValueType(1);
	SolverValueType alpha = SolverValueType(0);
	SolverValueType omega = SolverValueType(1);
	SolverValueType rho1;

	SolverVector r0(n);
	SolverVector r(n);
	SolverVector u(n,0);
	SolverVector xx(n);
	SolverVector Pv(n);

	std::vector<SolverVector> rr(L+1);
	std::vector<SolverVector> uu(L+1);

	for(int k = 0; k <= L; k++) {
		rr[k].resize(n, 0);
		uu[k].resize(n, 0);
	}

	SolverValueType tao[L+1][L+1];
	SolverValueType gamma[L+2];
	SolverValueType gamma_prime[L+2];
	SolverValueType gamma_primeprime[L+2];
	SolverValueType sigma[L+2];

	// r0 <- b - A * x
	spmv(x, r0);
	cusp::blas::axpby(b, r0, r0, SolverValueType(1), SolverValueType(-1));

	// r <- r0
	cusp::blas::copy(r0, r);

	// uu(0) <- u
	// rr(0) <- r
	// xx <- x
	thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(u.begin(), x.begin(), r.begin())), 
	             thrust::make_zip_iterator(thrust::make_tuple(u.end(), x.end(), r.end())), 
	             thrust::make_zip_iterator(thrust::make_tuple(uu[0].begin(), xx.begin(), rr[0].begin())));

	PrecVector left(n), right(n);

	while(!monitor.finished(r)) {

		rho0 = -omega * rho0;

		monitor.increment(0.25f);

		for(int j = 0; j < L; j++) {
			rho1 = cusp::blas::dotc(rr[j], r0);

			// return with failure
			if(rho0 == 0)
				return;

			SolverValueType beta = alpha * rho1 / rho0;
			rho0 = rho1;

			for(int i = 0; i <= j; i++) {
				// uu(i) = rr(i) - beta * uu(i)
				cusp::blas::axpby(rr[i], uu[i], uu[i], SolverValueType(1), -beta);
			}

			// uu(j+1) <- A * P^(-1) * uu(j);
			{
				cusp::blas::copy(uu[j], left);
				precond.solve(left, right);
				cusp::blas::copy(right, Pv);
			}
			spmv(Pv, uu[j+1]);

			// gamma <- uu(j+1) . r0;
			SolverValueType gamma = cusp::blas::dotc(uu[j+1], r0);
			if(gamma == 0)
				return;

			alpha = rho0 / gamma;

			for(int i = 0; i <= j; i++) {
				// rr(i) <- rr(i) - alpha * uu(i+1)
				cusp::blas::axpy(uu[i+1], rr[i], SolverValueType(-alpha));
			}

			// rr(j+1) = A * P^(-1) * rr(j)
			{
				cusp::blas::copy(rr[j], left);
				precond.solve(left, right);
				cusp::blas::copy(right, Pv);
			}
			spmv(Pv, rr[j+1]);
			
			// xx <- xx + alpha * uu(0)
			cusp::blas::axpy(uu[0], xx, alpha);

			if(monitor.finished(rr[0])) {
				cusp::blas::copy(xx, left);
				precond.solve(left, right);
				cusp::blas::copy(right, x);
				return;
			}
		}


		for(int j = 1; j <= L; j++) {
			for(int i = 1; i < j; i++) {
				tao[i][j] = cusp::blas::dotc(rr[j], rr[i]) / sigma[i];
				cusp::blas::axpy(rr[i], rr[j], -tao[i][j]);
			}
			sigma[j] = cusp::blas::dotc(rr[j], rr[j]);
			if(sigma[j] == 0)
				return;
			gamma_prime[j] = cusp::blas::dotc(rr[j], rr[0]) / sigma[j];
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
		cusp::blas::axpy(rr[0], xx,    gamma[1]);
		cusp::blas::axpy(rr[L], rr[0], -gamma_prime[L]);
		cusp::blas::axpy(uu[L], uu[0], -gamma[L]);

		monitor.increment(0.25f);

		if (monitor.finished(rr[0])) {
			cusp::blas::copy(xx, left);
			precond.solve(left, right);
			cusp::blas::copy(right, x);
			return;
		}

		monitor.increment(0.25f);

		// uu(0) <- uu(0) - sum_j { gamma(j) * uu(j) }
		// xx    <- xx    + sum_j { gamma''(j) * rr(j) }
		// rr(0) <- rr(0) - sum_j { gamma'(j) * rr(j) }
		for(int j = 1; j < L; j++) {
			cusp::blas::axpy(uu[j], uu[0],  -gamma[j]);
			cusp::blas::axpy(rr[j], xx,     gamma_primeprime[j]);
			cusp::blas::axpy(rr[j], rr[0],  -gamma_prime[j]);

			if (monitor.finished(rr[0])) {
				cusp::blas::copy(xx, left);
				precond.solve(left, right);
				cusp::blas::copy(right, x);
				return;
			}
		}

		// u <- uu(0)
		// x <- xx
		// r <- rr(0)
		thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(uu[0].begin(), xx.begin(), rr[0].begin())), 
		             thrust::make_zip_iterator(thrust::make_tuple(uu[0].end(), xx.end(), rr[0].end())), 
		             thrust::make_zip_iterator(thrust::make_tuple(u.begin(), x.begin(), r.begin())));

		monitor.increment(0.25f);
	}
}

/// Specializations of the generic spike::bicgstabl function for L=1
template <typename SpmvOperator, typename SolverVector, typename PrecVector>
void bicgstab (SpmvOperator&                       spmv,
               SolverVector&                       x,
               const SolverVector&                 b,
               Monitor<SolverVector>&              monitor,
               Precond<PrecVector>&                precond)
{
	bicgstabl<SpmvOperator, SolverVector, PrecVector, 1>(spmv, x, b, monitor, precond);
}

/// Specializations of the generic spike::bicgstabl function for L=2
template <typename SpmvOperator, typename SolverVector, typename PrecVector>
void bicgstab2(SpmvOperator&                       spmv,
               SolverVector&                       x,
               const SolverVector&                 b,
               Monitor<SolverVector>&              monitor,
               Precond<PrecVector>&                precond)
{
	bicgstabl<SpmvOperator, SolverVector, PrecVector, 2>(spmv, x, b, monitor, precond);
}


/// Preconditioned CG Krylov method
/**
 * \tparam SpmvOperator is a functor class for sparse matrix-vector product.
 * \tparam SolverVector is the vector type for the linear system solution.
 * \tparam PrecVector is the vector type used in the preconditioner.
 */
template <typename SpmvOperator, typename SolverVector, typename PrecVector>
void cg(SpmvOperator&                       spmv,
        SolverVector&                       x,
        const SolverVector&                 b,
        Monitor<SolverVector>&              monitor,
        Precond<PrecVector>&                precond)
{
	typedef typename SolverVector::value_type   SolverValueType;
	typedef typename SolverVector::memory_space MemorySpace;

	typedef typename cusp::array1d<int, MemorySpace>  IntVector;

	// Allocate workspace
	int  n = b.size();

	SolverVector r(n);

	// r <- b - A * x
	spmv(x, r);
	cusp::blas::axpby(b, r, r, SolverValueType(1), SolverValueType(-1));

	if (monitor.finished(r))
		return;

	SolverVector z(n), p(n);
	PrecVector r_prec(n), z_prec(n);

	{
		cusp::blas::copy(r, r_prec);
		precond.solve(r_prec, z_prec);
		cusp::blas::copy(z_prec, z);
	}

	// p <- z
	cusp::blas::copy(z, p);

	SolverValueType alpha, beta, dot_rz;

	SolverVector Ap(n);


	while(true) {

		spmv(p, Ap);

		dot_rz = cusp::blas::dotc(r, z);

		alpha = dot_rz / cusp::blas::dotc(p, Ap);

		cusp::blas::axpy(p, x, SolverValueType(alpha));

		cusp::blas::axpy(Ap, r, SolverValueType(-alpha));

		monitor.increment(1.f);
		if (monitor.finished(r))
			return;

		{
			cusp::blas::copy(r, r_prec);
			precond.solve(r_prec, z_prec);
			cusp::blas::copy(z_prec, z);
		}

		beta = cusp::blas::dotc(r, z) / dot_rz;
		cusp::blas::axpby(z, p, p, SolverValueType(1), SolverValueType(beta));
	}
}



} // namespace spike



#endif

