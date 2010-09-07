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
methods related to multiscalar neighbourhood density estimation. It is a C
library that takes sets of geographic data as inputs and computes a
smoothed representation of this data taking account of neighbourhood's
influence.

For more information: http://hyantes.gforge.inria.fr


Here we present 3 different executions of the same C sequential code:

- sequential execution

- automatic parallelization with p4a for parallel execution on multicores
  with OpenMP

- automatic parallelization with p4a --cuda for parallel execution on
  nVidia GPU


There is a simple Makefile used to launch the different phases.

You need to have gnuplot installed to be able to display the results.

For the sequential execution

  make seq : build the sequential program from hyantes-static-99.c

  make run_seq : build first if needed, then run the sequential program

  make display_seq : build first if needed, run if needed, then display
  the output of the sequential version with gnuplot

For the OpenMP parallel execution on multicores:

  make openmp : parallelize the code to OpenMP source
  hyantes-static-99.p4a.c and program hyantes-static-99_openmp

  make run_openmp : build first if needed, then run the OpenMP parallel
  program

  make display_openmp : build first if needed, run if needed, then display
  the output of the OpenMP parallel version with gnuplot

For the CUDA parallel execution on nVidia GPU:

  make cuda : parallelize the code to CUDA source hyantes-static-99.p4a.cu
  and program hyantes-static-99_cuda

  make run_cuda : build first if needed, then run the CUDA parallel
  program

    Do not forget to have the CUDA runtime correctly
    installed. LD_LIBRARY_PATH should contain at least the location of
    CUDA runtime library.

  make display_cuda : build first if needed, run if needed, then display
  the output of the CUDA parallel version with gnuplot

You can set the P4A_OPTIONS variable to pass some options to p4a.

  For example, globally with an:
  export P4A_OPTIONS='--nvcc-flags="-gencode arch=compute_20,code=sm_20 -DP4A_DEBUG"'
  or locally with:
  make P4A_OPTIONS='--nvcc-flags="-DP4A_DEBUG"' run_cuda


For a full demo:

  make demo : to chain make display_seq, make display_openmp, make display_cuda


Some results:

  We measure the wall-clock time that includes startup time, data load time
  and output write time, that is the real time understood by users. By
  measuring kernel time only, speed-up would be better but less
  representative of the real application (Amdahl...).

  On one of our WildNode with 2 Intel Xeon X5670 @ 2.93GHz (12 cores) and
  a Tesla C2050 (Fermi), Linux/Ubuntu 10.04, gcc 4.4.3, CUDA 3.1, we
  measure in production:

  - Sequential execution time on CPU: 30.355s

  - OpenMP parallel execution time on CPUs: 3.859s, speed-up: 7.87

  - CUDA parallel execution time on GPU: 0.441s, speed-up: 68.8
