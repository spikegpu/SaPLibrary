/** \file bicgstab.h
 *  \brief BiCGStab preconditioned iterative Krylov solver.
 */

/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 * Modified CUSP bicgstab
 * (Radu Serban, 2014)
 */

#ifndef SPIKE_BICGSTAB_H
#define SPIKE_BICGSTAB_H

#include <cusp/array1d.h>
#ifdef   USE_OLD_CUSP
#include <cusp/blas.h>
#else
#include <cusp/blas/blas.h>
#endif
#include <cusp/multiply.h>

#include <spike/monitor.h>


namespace spike {

/// Preconditioned BiCGStab Krylov method
/**
 * \tparam LinearOperator is a functor class for sparse matrix-vector product.
 * \tparam Vector is the vector type for the linear system solution.
 * \tparam Monitor is the convergence test object.
 * \tparam Preconditioner is the preconditioner.
 */
template <typename LinearOperator, typename Vector, typename Monitor, typename Preconditioner>
void bicgstab(LinearOperator&  A,
              Vector&          x,
              Vector&          b,
              Monitor&         monitor,
              Preconditioner&  M)
{
	typedef typename Vector::value_type   ValueType;
	typedef typename Vector::memory_space MemorySpace;

	const size_t N = A.num_rows;

	// allocate workspace
	cusp::array1d<ValueType,MemorySpace> y(N);

	cusp::array1d<ValueType,MemorySpace>   p(N);
	cusp::array1d<ValueType,MemorySpace>   r(N);
	cusp::array1d<ValueType,MemorySpace>   r_star(N);
	cusp::array1d<ValueType,MemorySpace>   s(N);
	cusp::array1d<ValueType,MemorySpace>  Mp(N);
	cusp::array1d<ValueType,MemorySpace> AMp(N);
	cusp::array1d<ValueType,MemorySpace>  Ms(N);
	cusp::array1d<ValueType,MemorySpace> AMs(N);

	// y <- Ax
	cusp::multiply(A, x, y);

	// r <- b - A*x
	cusp::blas::axpby(b, y, r, ValueType(1), ValueType(-1));

	// p <- r
	cusp::blas::copy(r, p);

	// r_star <- r
	cusp::blas::copy(r, r_star);

	ValueType r_r_star_old = cusp::blas::dotc(r_star, r);

	while (!monitor.finished(r)) {
		// Prevent divison by zero at this iteration.
		if (r_r_star_old == 0) {
			monitor.stop(-10, "r_r_star is zero");
			break;
		}

		// Mp = M*p
		cusp::multiply(M, p, Mp);

		// AMp = A*Mp
		cusp::multiply(A, Mp, AMp);

		// alpha = (r_j, r_star) / (A*M*p, r_star)
		ValueType tmp1 = cusp::blas::dotc(r_star, AMp);
		if (tmp1 == 0) {
			monitor.stop(-11, "r_star * AMp is zero");
			break;
		}
		ValueType alpha = r_r_star_old / tmp1;

		// s_j = r_j - alpha * AMp
		cusp::blas::axpby(r, AMp, s, ValueType(1), ValueType(-alpha));

		if (monitor.finished(s)){
			// x += alpha*M*p_j
			cusp::blas::axpby(x, Mp, x, ValueType(1), ValueType(alpha));
			break;
		}

		// Ms = M*s_j
		cusp::multiply(M, s, Ms);

		// AMs = A*Ms
		cusp::multiply(A, Ms, AMs);

		// omega = (AMs, s) / (AMs, AMs)
		ValueType tmp2 = cusp::blas::dotc(AMs, AMs);
		if (tmp2 == 0) {
			monitor.stop(-12, "AMs * AMs is zero");
			break;
		}
		ValueType omega = cusp::blas::dotc(AMs, s) / tmp2;

		// x_{j+1} = x_j + alpha*M*p_j + omega*M*s_j
		cusp::blas::axpbypcz(x, Mp, Ms, x, ValueType(1), alpha, omega);

		// r_{j+1} = s_j - omega*A*M*s
		cusp::blas::axpby(s, AMs, r, ValueType(1), -omega);

		// beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)
		ValueType r_r_star_new = cusp::blas::dotc(r_star, r);

		ValueType beta = (r_r_star_new / r_r_star_old) * (alpha / omega);
		r_r_star_old = r_r_star_new;

		// p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
		cusp::blas::axpbypcz(r, p, AMp, p, ValueType(1), beta, -beta*omega);

		++monitor;
	}
}


} // end namespace spike



#endif
