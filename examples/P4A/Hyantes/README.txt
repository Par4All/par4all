If you do not have write access to this directory, copy its content (with
cp -a for example) somewhere you can write into and try the examples from
this new location.

Hyantes is a library to compute neighbourhood population potential with
scale control. It is developed by the Mescal team from the Laboratoire
Informatique de Grenoble (France), as a part of Hypercarte project. The
Hypercarte project aims to develop new methods for the cartographic
representation of human distributions (population density, population
increase, etc.) with various smoothing functions and opportunities for
time-scale animations of maps. Hyantes provides one of the smoothing
methods related to multi-scalar neighbourhood density estimation. It is a C
library that takes sets of geographic data as inputs and computes a
smoothed representation of this data taking account of neighbourhood's
influence.

For more information: http://hyantes.gforge.inria.fr


Here we present 5 different executions of the same C sequential code:

- sequential execution

- automatic parallelization with p4a for parallel execution on multi-cores
  with OpenMP

- automatic parallelization with p4a --cuda for parallel execution on
  nVidia GPU

- automatic parallelization with p4a --accel for an OpenMP parallel 
  emulation of a GPU-like accelerator

- automatic parallelization with p4a --opencl the OpenCL parallel 
  execution on nVidia GPU

You need to have gnuplot installed to be able to display the results.

See ../README.txt to get the generic commands to parallelize, build and
execute this example. More commands allowing to display the computation
results using gnuplot, are the following:

For the sequential execution

  make display_seq : build first if needed, run if needed, then display
  		the output of the sequential version with gnuplot

For the OpenMP parallel execution on multi-cores:

  make display_openmp : build first if needed, run if needed, then display
  		the output of the OpenMP parallel version with gnuplot

For the CUDA parallel execution on nVidia GPU:

  make display_cuda : build first if needed, run if needed, then display
  		the output of the CUDA parallel version with gnuplot

For an OpenMP parallel emulation of a GPU-like accelerator (useful for
debugging, without any GPU):

  make display_accel : build first if needed, run if needed, then display
  		the output of the accelerator OpenMP parallel emulation
		version with gnuplot

For the OpenCL parallel execution on nVidia GPU:

  make display_opencl : build first if needed, run if needed, then display
  		the output of the OpenCL parallel version with gnuplot

For a full demo:

  make demo : to chain make display_seq, display_openmp, display_cuda,
  		display_accel-openmp, display_opencl


To run this example on GPU that does not support double precision, you
should compile it with
make USE_FLOAT=1 P4A_OPTIONS='--cuda-cc=1.1' ...
or the results are just garbage (because if nvcc has a single precision
fall-back, the communication correctly computed by Par4All are still
in... double). Of course the results are slightly different in single
precision compared to double precision.




Some results:

  We measure the wall-clock time that includes startup time, data load time
  and output write time, that is the real time understood by users. By
  measuring kernel time only, speed-up would be better but less
  representative of the real application (Amdahl...).

  On one of our WildNode with 2 Intel Xeon X5670 @ 2.93GHz (12 cores) and
  a nVidia Tesla C2050 (Fermi), Linux/Ubuntu 10.04, gcc 4.4.3, CUDA 3.1,
  we measure in production:

  - Sequential execution time on CPU: 30.355s

  - OpenMP parallel execution time on CPUs: 3.859s, speed-up: 7.87

  - CUDA parallel execution time on GPU: 0.441s, speed-up: 68.8

  For *single precision* (compiled with make USE_FLOAT=1) on a HP
  EliteBook 8730w laptop (with an Intel Core2 Extreme Q9300 @ 2.53GHz (4
  cores) and a nVidia GPU Quadro FX 3700M, 16 multiprocessors, 128 cores,
  architecture 1.1) with Linux/Debian/sid, gcc 4.4.3, CUDA 3.1:

  - Sequential execution time on CPU: 38s

  - OpenMP parallel execution time on CPUs: 18.9s, speed-up: 2.01

  - CUDA parallel execution time on GPU: 1.57s, speed-up: 24.2



### Local Variables:
### mode: flyspell
### ispell-local-dictionary: "american"
### End:
