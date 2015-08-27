/** \file minres.h
 *  \brief MINRES preconditioned iterative solver.
 */

#ifndef SAP_MINRES_H
#define SAP_MINRES_H

#include <vector>

#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif
#include <cusp/multiply.h>
#include <cusp/array1d.h>

#include <sap/monitor.h>
#include <sap/precond.h>


namespace sap {

// Additional stopping conditions, specific to MINRES:
//   code =  10     beta2 = 0.  If M = I, b and x are eigenvectors
//   code =  11     x has converged to an eigenvector
//   code =  12     Reasonable accuracy achieved, given eps
//   code =  13     A least-squares solution was found, given tol 
//   code = -10     The preconditioner is not positive definite

/// Preconditioned MINRES method
/**
 * \tparam LinearOperator is a functor class for sparse matrix-vector product.
 * \tparam Vector is the vector type for the linear system solution.
 * \tparam Monitor is the convergence test object.
 * \tparam Preconditioner is the preconditioner (must be positive definite)
 */
template <typename LinearOperator, typename Vector, typename Monitor, typename Preconditioner>
void minres(LinearOperator&  A,
            Vector&          x,
            const Vector&    b,
            Monitor&         monitor,
            Preconditioner&  P)
{
	typedef typename Vector::value_type   ValueType;
	typedef typename Vector::memory_space MemorySpace;

	ValueType eps = std::numeric_limits<ValueType>::epsilon();
	////ValueType shift = 0;

	int  n = b.size();

	// Set up y and v for the first Lanczos vector v1.
	cusp::array1d<ValueType,MemorySpace>  y(n);
	cusp::array1d<ValueType,MemorySpace>  r1(n);

	cusp::multiply(A, x, r1);
	////cusp::blas::axpby(r1, x, r1, ValueType(1), -shift);
	cusp::blas::axpby(b, r1, r1, ValueType(1), ValueType(-1));

	cusp::multiply(P, r1, y);

	ValueType beta1 = cusp::blas::dotc(r1, y);

	// Test for indefinite preconditioner.
	//    beta1 < 0  --> P is not positive definite.
	//    beta1 = 0  --> will stop later with x=x0
	//    beta1 > 0  --> normalize to get v1 later
	if (beta1 < 0) {
		monitor.stop(-10, "The preconditioner is not positive definite");
		beta1 *= -1;      // ensure we do not incorrectly report convergence
	} else if (beta1 > 0) {
		beta1 = std::sqrt(beta1);
	}

	// Initialize other quantities
	ValueType Anorm = 0;
	ValueType ynorm = 0;
	ValueType phibar = beta1;    // residual norm

	ValueType oldb(0), beta(beta1), dbar(0), epsln(0), oldeps(0);
	ValueType phi(0), rhs1(beta1);
	ValueType rhs2(0), tnorm2(0);
	ValueType cs(-1), sn(0);
	ValueType gmax(0), gmin(std::numeric_limits<ValueType>::max());
	ValueType alpha(0), gamma(0);
	ValueType delta(0), gbar(0);
	ValueType z(0);

	cusp::array1d<ValueType,MemorySpace>  v(n);
	cusp::array1d<ValueType,MemorySpace>  w(n, 0);
	cusp::array1d<ValueType,MemorySpace>  w1(n, 0);
	cusp::array1d<ValueType,MemorySpace>  w2(n, 0);
	cusp::array1d<ValueType,MemorySpace>  r2 = r1;

	// Main loop
	while (!monitor.finished(phibar)) {

		// Obtain quantities for the next Lanczos vector
		ValueType s = 1/beta;
		cusp::blas::copy(y, v);
		cusp::blas::scal(v, s);

		cusp::multiply(A, v, y);
		////cusp::blas::axpby(y, v, y, ValueType(1), -shift);
		if (monitor.iteration_count() > 0)
			cusp::blas::axpby(y, r1, y, ValueType(1), -beta/oldb);

		alpha = cusp::blas::dotc(v, y);
		cusp::blas::axpby(y, r2, y, ValueType(1), -alpha/beta);
		cusp::blas::copy(r2, r1);
		cusp::blas::copy(y, r2);
		cusp::multiply(P, r2, y);

		oldb = beta;
		beta = cusp::blas::dotc(r2, y);

		if (beta < 0) {
			// Preconditioner is not positive definite. Force failure.
			monitor.stop(-10, "The preconditioner is not positive definite");
			break;
		}

		beta = std::sqrt(beta);
		tnorm2 += alpha*alpha + oldb*oldb + beta*beta;

		if (monitor.iteration_count() == 0) {
			if (beta/beta1 <= 10 * eps)
				monitor.stop(10, "beta2 = 0. If M = I, b and x are eigenvectors");
			// Terminate later
		}

		// Apply previous rotation Q_{k-1}
		oldeps = epsln;
		delta  = cs*dbar + sn*alpha;
		gbar   = sn*dbar - cs*alpha;
		epsln  =           sn*beta;
		dbar   =         - cs*beta;

		ValueType root = std::sqrt(gbar*gbar + dbar*dbar);

		// Compute next plane rotation Q_k
		gamma = std::sqrt(gbar*gbar + beta*beta); // gamma_k
		gamma = std::max(gamma, eps);
		cs = gbar/gamma;                     // c_k
		sn = beta/gamma;                     // s_k
		phi = cs*phibar;                     // phi_k
		phibar = sn*phibar;                  // phibar_{k+1}

		// Update x
		ValueType denom = 1/gamma;
		cusp::blas::copy(w2, w1);
		cusp::blas::copy(w, w2);
		cusp::blas::axpbypcz(v, w1, w2, w, denom, -oldeps*denom, -delta*denom);
		cusp::blas::axpby(x, w, x, ValueType(1), phi);

		// Go round again
		gmax    = std::max(gmax, gamma);
		gmin    = std::min(gmin, gamma);
		z       = rhs1/gamma;
		rhs1    = rhs2 - delta*z;
		rhs2    =      - epsln*z;

		// Estimate various norms
		Anorm = std::sqrt(tnorm2);
		ynorm = cusp::blas::nrm2(x);

		// Additional checks that can trigger termination.
		if(monitor.getCode() != 10) {
			ValueType test2  = root/ Anorm;  // ||A r_{k-1}|| / (||A|| ||r_{k-1}||)
			ValueType Acond = gmax/gmin;     // Estimate cond(A)

			if(Acond >= ValueType(0.1)/eps)         monitor.stop(11, "x has converged to an eigenvector");
			if(Anorm*eps*ynorm >= beta1)            monitor.stop(12, "Reasonable accuracy achieved, given eps");
			if(test2 <= monitor.getAbsTolerance())  monitor.stop(13, "A least-squares solution was found, given tol");
		}

		++monitor;
	}

}



} // namespace sap



#endif

