Stars-pm is a particle-mesh N-body cosmological simulation, written by Dominique Aubert and Mehdi Amini.


Here we present 3 different executions of the same C sequential code:

- sequential execution

- automatic parallelization with p4a for parallel execution on multicores
  with OpenMP

- automatic parallelization with p4a --cuda for parallel execution on
  nVidia GPU


There is a simple Makefile used to launch the different phases.

You need to have fftw3f library installed to be able to link the code, and 
optionnaly opengl and/or GTK for visualisation.

For the sequential execution

  make seq : build the sequential program from hyantes-static-99.c

  make run-seq : build first if needed, then run the sequential program

For the OpenMP parallel execution on multicores: 

  (same as previously with -openmp instead of -seq)

For the CUDA parallel execution on nVidia GPU:

  (same as previously with -cuda instead of -seq or -openmp)


To get an output you might add opengl=1 and/or gtk=1 on cmd line, for instance 
"make run-seq opengl=1". You might also run "make clean" first to force rebuilding.


You can set the P4A_OPTIONS variable to pass some options to p4a.

  For example, globally with an:
  export P4A_OPTIONS='--nvcc-flags="-gencode arch=compute_20,code=sm_20 -DP4A_DEBUG"'
  or locally with:
  make P4A_OPTIONS='--nvcc-flags="-DP4A_DEBUG"' run_cuda


To run this example on GPU that does not support double precision, you
should compile it with make USE_FLOAT=1 or the results are just garbage
(because if nvcc has a single precision fall-back, the communication
correctly computed by Par4All are still in... double). Of course the
results are slightly different in single precision compared to double
precision.

Some results:

  We measure the wall-clock time that includes startup time, data load time
  and output write time, that is the real time understood by users. By
  measuring kernel time only, speed-up would be better but less
  representative of the real application (Amdahl...).

  On one of our WildNode with 2 Intel Xeon X5670 @ 2.93GHz (12 cores) and
  a nVidia Tesla C2050 (Fermi), Linux/Ubuntu 10.04, gcc 4.4.3, CUDA 3.1,
  we measure in production:

  - Sequential execution time on CPU: s

  - OpenMP parallel execution time on CPUs: s, speed-up: 

  - CUDA parallel execution time on GPU: s, speed-up: 

  For *single precision* (compiled with make USE_FLOAT=1) on a HP
  EliteBook 8730w laptop (with an Intel Core2 Extreme Q9300 @ 2.53GHz (4
  cores) and a nVidia GPU Quadro FX 3700M, 16 multiprocessors, 128 cores,
  architecture 1.1) with Linux/Debian/sid, gcc 4.4.3, CUDA 3.1:

  - Sequential execution time on CPU: s

  - OpenMP parallel execution time on CPUs: s, speed-up: 

  - CUDA parallel execution time on GPU: s, speed-up: 

