/** \file minres.h
 *  \brief MINRES preconditioned iterative solver.
 */

#ifndef SPIKE_MINRES_H
#define SPIKE_MINRES_H

#include <vector>

#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/array1d.h>

#include <spike/monitor.h>
#include <spike/precond.h>


namespace spike {

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

	// Initialize stopping criteria
	bool done = false;
	int istop = 0;
	int itn = 0;

	ValueType Anorm = 0;
	ValueType rnorm = 0;
	ValueType ynorm = 0;

	// Set up y and v for the first Lanczos vector v1.
	Vector y(n);
	Vector r1(n);

	cusp::multiply(A, x, r1);
	////cusp::blas::axpby(r1, x, r1, ValueType(1), -shift);
	cusp::blas::axpby(b, r1, r1, ValueType(1), ValueType(-1));

	cusp::multiply(P, r1, y);

	ValueType beta1 = cusp::blas::dotc(r1, y);

	// Test for indefinite preconditioner
	if (beta1 < 0) {
		// P is not positive definite
		istop = 9;
		done = true;
	} else if (beta1 == 0) {
		// stop with x = x0
		done = true;
	} else {
		// normalize y to get v1 later
		beta1 = std::sqrt(beta1);
	}

	// Initialize other quantities
	ValueType oldb(0), beta(beta1), dbar(0), epsln(0), oldeps(0);
	ValueType phi(0), phibar(beta1), rhs1(beta1);
	ValueType rhs2(0), tnorm2(0);
	ValueType cs(-1), sn(0);
	ValueType gmax(0), gmin(std::numeric_limits<ValueType>::max());
	ValueType alpha(0), gamma(0);
	ValueType delta(0), gbar(0);
	ValueType z(0);

	Vector v(n);
	Vector w(n, 0);
	Vector w1(n, 0);
	Vector w2(n, 0);
	Vector r2 = r1;

	// Main loop
	int max_iter = monitor.getMaxIterations();
	ValueType tol = monitor.getTolerance();

	if (!done) {
		for (itn = 0; itn < max_iter; itn++) {

			// Obtain quantities for the next Lanczos vector
			ValueType s = 1/beta;
			cusp::blas::copy(y, v);
			cusp::blas::scal(v, s);

			cusp::multiply(A, v, y);
			////cusp::blas::axpby(y, v, y, ValueType(1), -shift);
			if (itn)
				cusp::blas::axpby(y, r1, y, ValueType(1), -beta/oldb);

			alpha = cusp::blas::dotc(v, y);
			cusp::blas::axpby(y, r2, y, ValueType(1), -alpha/beta);
			cusp::blas::copy(r2, r1);
			cusp::blas::copy(y, r2);
			cusp::multiply(P, r2, y);

			oldb = beta;
			beta = cusp::blas::dotc(r2, y);

			if (beta < 0) {
				istop = 9;
				break;
			}

			beta = std::sqrt(beta);
			tnorm2 += alpha*alpha + oldb*oldb + beta*beta;

			if (itn == 0) {
				if (beta/beta1 <= 10 * eps)
					istop = 10;    // terminate later
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
			cusp::blas::axpby(w1, w2, w, -oldeps, -delta);
			cusp::blas::axpby(v, w, w, denom, denom);
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
			rnorm = phibar;

			// Check stopping criteria
			if(0 == istop) {
				//// Radu:  changed the stopping criteria to be consistent with 
				////       the other solvers.  That is, check relative residual.
				//ValueType test1  = rnorm/(Anorm*ynorm);           // ||r||/(||A|| ||x||)
				ValueType test1  = rnorm / monitor.getRHSNorm();  // ||r|| / ||b||
				ValueType test2  = root/ Anorm;                   // ||A r_{k-1}|| / (||A|| ||r_{k-1}||)

				//This test work if tol < eps
				ValueType t1 = ValueType(1) + test1;
				ValueType t2 = ValueType(1) + test2;
				if(t2 <= 1) istop = 2;
				if(t1 <= 1) istop = 1;

				// Estimate cond(A)
				ValueType Acond = gmax/gmin;

				if(itn >= max_iter-1) istop = 6;
				if(Acond >= ValueType(0.1)/eps) istop = 4;
				if(Anorm*eps*ynorm >= beta1)   istop = 3;
				if(test2 <= tol  )  istop = 2;
				if(test1 <= tol)    istop = 1;
			}

			if(0 != istop)
				break;
		}
	}

	// HACK
	monitor.increment(itn);
	monitor.finished(rnorm);

}



} // namespace spike



#endif

