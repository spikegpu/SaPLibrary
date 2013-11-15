SpikeGPU Library version 1.0.0
==============================

SpikeGPU is a C++ template library which provides a SPIKE-based preconditioner for the solution of large-scale sparse linear solvers using Krylov-space iterative solvers on CUDA architecture GPUs. 

The SpikeGPU library is built on top of CUSP and Thrust. 

Additional information available at http://spikegpu.sbel.org.

Directory structure
-------------------
There are two top-level directories:
<dl>
  <dt>spike/</dt>     
    <dd>contains the library's header files.</dd>
  <dt>examples/</dt>
    <dd>provides several example programs using the SpikeGPU solver. <dd>
</dl>

Dependencies
------------
SpikeGPU requires CUDA and the CUSP library, available from https://github.com/cusplibrary.

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
  spike::Solver<Vector, float> spikeGPU(10, options);
  spike::SpmvCusp<Matrix>      spmv(A);
  
  // Set the solution initial guess to zero.
  Vector x(A.num_rows, 0.0);
  
  // Solve the problem.
  spikeGPU.setup(A);
  bool success = spikeGPU.solve(spmv, b, x);
  
  // Extract solver statistics.
  spike::Stats stats = spikeGPU.getStats();
  
  // ...
}
```

Building and running the example drivers
----------------------------------------
Use CMake to configure the provided example drivers:
* driver_mm  - sample program for using SpikeGPU with a matrix read from a Matrix Market file.
* driver_seq - sample program illustrating the use of SpikeGPU on a sequence of matrices with the same sparsity pattern.
* driver_views - sample program illustrating the use of SpikeGPU with CUSP array views.
* driver_banded - sample program illustrating the use of SpikeGPU to solve banded systems.

To see a full list of the arguments for driver_mm as an example, use
`driver_mm -h`

Support
-------
Submit bug reports and feature requests at https://github.com/spikegpu/SpikeLibrary/issues.

Feel free to fork the github repository and submit pull requests.

License
-------
The code is available from https://github.com/spikegpu/SpikeLibrary under a BSD-3 license. See the file LICENSE.


