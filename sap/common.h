/** \file common.h
 *  \brief Definition of commonly used macros and solver configuration constants.
 */

#ifndef SAP_COMMON_H
#define SAP_COMMON_H



#define ALWAYS_ASSERT

#ifdef WIN32
typedef long long int64_t;
#endif

#include <cusp/version.h>

/**
 * If ALWAYS_ASSERT is defined, we make sure that  assertions are triggered even if NDEBUG is defined.
 */
#ifdef ALWAYS_ASSERT
// If NDEBUG is actually defined, remember this so
// we can restore it.
#  ifdef NDEBUG
#    define NDEBUG_ACTIVE
#    undef NDEBUG
#  endif
// Include the assert.h header file here so that it can
// do its stuff while NDEBUG is guaranteed to be disabled.
#  include <assert.h>
// Restore NDEBUG mode if it was active.
#  ifdef NDEBUG_ACTIVE
#    define NDEBUG
#    undef NDEBUG_ACTIVE
#  endif
#else
// Include the assert.h header file using whatever the
// current definition of NDEBUG is.
#  include <assert.h>
#endif


// ----------------------------------------------------------------------------


#define BURST_VALUE (1e-7)
#define BURST_NEW_VALUE (1e-4)
#define MATRIX_MUL_BLOCK_SIZE (16)
#define MAT_VEC_MUL_BLOCK_SIZE (16)


#if CUSP_VERSION < 500
#  define USE_OLD_CUSP
#else
#  ifdef USE_OLD_CUSP
#    undef USE_OLD_CUSP
#  endif
#endif

namespace sap {

const unsigned int BLOCK_SIZE = 512;

const unsigned int MAX_GRID_DIMENSION = 32768;

const unsigned int CRITICAL_THRESHOLD = 70;

/**
 * This defines the types of Krylov subspace methods used in SaP.
 */
enum KrylovSolverType {
	// CUSP solvers
	BiCGStab_C,
	GMRES_C,
	CG_C,
	CR_C,
	// SAP solvers
	BiCGStab1,
	BiCGStab2,
	BiCGStab,
	MINRES
};

enum FactorizationMethod {
	LU_UL,
	LU_only
};

/**
 * This defines the types of SaP preconditioners.
 */
enum PreconditionerType {
	Spike,
	Block,
	None
};

inline
void kernelConfigAdjust(int &numThreads, int &numBlockX, const int numThreadsMax) {
	if (numThreads > numThreadsMax) {
		numBlockX = (numThreads + numThreadsMax - 1) / numThreadsMax;
		numThreads = numThreadsMax;
	}
}

inline
void kernelConfigAdjust(int &numThreads, int &numBlockX, int &numBlockY, const int numThreadsMax, const int numBlockXMax) {
	kernelConfigAdjust(numThreads, numBlockX, numThreadsMax);
	if (numBlockX > numBlockXMax) {
		numBlockY = (numBlockX + numBlockXMax - 1) / numBlockXMax;
		numBlockX = numBlockXMax;
	}
}


} // namespace sap


#endif
