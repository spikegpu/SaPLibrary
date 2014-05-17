/** \file bicgstab2.h
 *  \brief BiCGStab(L) preconditioned iterative Krylov solver.
 */

#ifndef SPIKE_BICGSTAB_2_H
#define SPIKE_BICGSTAB_2_H

#include <vector>

#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/array1d.h>

#include <spike/monitor.h>


namespace spike {

/// Preconditioned BiCGStab(L) Krylov method
/**
 * \tparam LinearOperator is a functor class for sparse matrix-vector product.
 * \tparam Vector is the vector type for the linear system solution.
 * \tparam Monitor is the convergence test object.
 * \tparam Preconditioner is the preconditioner.
 * \tparam L is the degree of the BiCGStab(L) method.
 */
template <typename LinearOperator, typename Vector, typename Monitor, typename Preconditioner, int L>
void bicgstabl(LinearOperator&  A,
               Vector&          x,
               const Vector&    b,
               Monitor&         monitor,
               Preconditioner&  P)
{
	typedef typename Vector::value_type   ValueType;
	typedef typename Vector::memory_space MemorySpace;

	// Allocate workspace
	int  n = b.size();

	ValueType rho0  = ValueType(1);
	ValueType alpha = ValueType(0);
	ValueType omega = ValueType(1);
	ValueType rho1;

	cusp::array1d<ValueType,MemorySpace>  r0(n);
	cusp::array1d<ValueType,MemorySpace>  r(n);
	cusp::array1d<ValueType,MemorySpace>  u(n,0);
	cusp::array1d<ValueType,MemorySpace>  xx(n);
	cusp::array1d<ValueType,MemorySpace>  Pv(n);

	std::vector<cusp::array1d<ValueType,MemorySpace> >  rr(L+1);
	std::vector<cusp::array1d<ValueType,MemorySpace> >  uu(L+1);

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
	A(x,r0);
	////cusp::multiply(A, x, r0);
	cusp::blas::axpby(b, r0, r0, ValueType(1), ValueType(-1));

	// r <- r0
	cusp::blas::copy(r0, r);

	// uu(0) <- u
	// rr(0) <- r
	// xx <- x
	thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(u.begin(), x.begin(), r.begin())), 
	             thrust::make_zip_iterator(thrust::make_tuple(u.end(), x.end(), r.end())), 
	             thrust::make_zip_iterator(thrust::make_tuple(uu[0].begin(), xx.begin(), rr[0].begin())));

	while(!monitor.finished(r)) {

		rho0 = -omega * rho0;

		monitor.increment(0.25f);

		for(int j = 0; j < L; j++) {
			rho1 = cusp::blas::dotc(rr[j], r0);

			// return with failure
			if(rho0 == 0) {
				monitor.stop(-10, "rho0 is zero");
				return;
			}

			ValueType beta = alpha * rho1 / rho0;
			rho0 = rho1;

			for(int i = 0; i <= j; i++) {
				// uu(i) = rr(i) - beta * uu(i)
				cusp::blas::axpby(rr[i], uu[i], uu[i], ValueType(1), -beta);
			}

			// uu(j+1) <- A * P^(-1) * uu(j);
			cusp::multiply(P, uu[j], Pv);
			cusp::multiply(A, Pv, uu[j+1]);

			// gamma <- uu(j+1) . r0;
			ValueType gamma = cusp::blas::dotc(uu[j+1], r0);

			if(gamma == 0) {
				monitor.stop(-11, "gamma is zero");
				return;
			}

			alpha = rho0 / gamma;

			for(int i = 0; i <= j; i++) {
				// rr(i) <- rr(i) - alpha * uu(i+1)
				cusp::blas::axpy(uu[i+1], rr[i], ValueType(-alpha));
			}

			// rr(j+1) = A * P^(-1) * rr(j)
			cusp::multiply(P, rr[j], Pv);
			cusp::multiply(A, Pv, rr[j+1]);
			
			// xx <- xx + alpha * uu(0)
			cusp::blas::axpy(uu[0], xx, alpha);

			if(monitor.finished(rr[0])) {
				cusp::multiply(P, xx, x);
				return;
			}
		}


		for(int j = 1; j <= L; j++) {
			for(int i = 1; i < j; i++) {
				tao[i][j] = cusp::blas::dotc(rr[j], rr[i]) / sigma[i];
				cusp::blas::axpy(rr[i], rr[j], -tao[i][j]);
			}
			sigma[j] = cusp::blas::dotc(rr[j], rr[j]);
			if(sigma[j] == 0) {
				monitor.stop(-12, "a sigma value is zero");
				return;
			}
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
			cusp::multiply(P, xx, x);
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
				cusp::multiply(P, xx, x);
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
template <typename LinearOperator, typename Vector, typename Monitor, typename Preconditioner>
void bicgstab1(LinearOperator&  A,
               Vector&          x,
               const Vector&    b,
               Monitor&         monitor,
               Preconditioner&  P)
{
	bicgstabl<LinearOperator, Vector, Monitor, Preconditioner, 1>(A, x, b, monitor, P);
}

/// Specializations of the generic spike::bicgstabl function for L=2
template <typename LinearOperator, typename Vector, typename Monitor, typename Preconditioner>
void bicgstab2(LinearOperator&  A,
               Vector&          x,
               const Vector&    b,
               Monitor&         monitor,
               Preconditioner&  P)
{
	bicgstabl<LinearOperator, Vector, Monitor, Preconditioner, 2>(A, x, b, monitor, P);
}



} // namespace spike



#endif

