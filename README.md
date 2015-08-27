SaPGPU Library version 1.0.0
==============================

SaPGPU is a C++ template library which provides a SPIKE-based preconditioner for the solution of large-scale sparse linear solvers using Krylov-space iterative solvers on CUDA architecture GPUs. 

The SaPGPU library is built on top of CUSP and Thrust. 

Additional information available at http://sapgpu.sbel.org.

Directory structure
-------------------
There are two top-level directories:
<dl>
  <dt>spike/</dt>     
    <dd>contains the library's header files.</dd>
  <dt>examples/</dt>
    <dd>provides several example programs using the SaPGPU solver. <dd>
</dl>

Dependencies
------------
SaPGPU requires CUDA and the CUSP library, available from https://github.com/cusplibrary.

Example Usage
-------------
```C++
#include <spike/solver.h>
#include <spike/spmv.h>
#include <spike/exception.h>

typedef typename cusp::csr_matrix<int, double, cusp::device_memory> Matrix;
typedef typename cusp::array1d<double, cusp::device_memory>         Vector;

int main(int argc, char** argv) 
{
  // ...
  
  // Read the matrix and right-hand side vector from disk files.
  Matrix A;
  Vector b;
  cusp::io::read_matrix_market_file(A, "matrix.mtx");
  cusp::io::read_matrix_market_file(b, "rhs.mtx");
  
  // Create the Spike solver object and the SPMV functor. In the solver constructor,
  // specify the number of partitions and a structure with optional inputs.
  spike::Options               options;
  spike::Solver<Vector, float> sapGPU(10, options);
  spike::SpmvCusp<Matrix>      spmv(A);
  
  // Set the solution initial guess to zero.
  Vector x(A.num_rows, 0.0);
  
  // Solve the problem.
  sapGPU.setup(A);
  bool success = sapGPU.solve(spmv, b, x);
  
  // Extract solver statistics.
  spike::Stats stats = sapGPU.getStats();
  
  // ...
}
```

Building and running the example drivers
----------------------------------------
Use CMake to configure the provided example drivers:
* driver_mm  - sample program for using SaPGPU with a matrix read from a Matrix Market file.
* driver_seq - sample program illustrating the use of SaPGPU on a sequence of matrices with the same sparsity pattern.
* driver_views - sample program illustrating the use of SaPGPU with CUSP array views.
* driver_banded - sample program illustrating the use of SaPGPU to solve banded systems.

To see a full list of the arguments for driver_mm as an example, use
`driver_mm -h`

Support
-------
Submit bug reports and feature requests at https://github.com/spikegpu/SpikeLibrary/issues.

Feel free to fork the github repository and submit pull requests.

License
-------
The code is available from https://github.com/spikegpu/SpikeLibrary under a BSD-3 license. See the file LICENSE.


