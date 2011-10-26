Jacobi is a simple relaxation on an image that is done T times (400 times in
this demo). The convolution kernel is the matrix
0   1/4  0
1/4  0  1/4
0   1/4  0

It uses the top left part of the HPC Project logo as input

See ../README.txt to get the generic commands to parallelize, build and execute 
this example. More commands allowing to display the computation results, 
are the following:

For the sequential execution

  make display_seq : build first if needed, run if needed, then display
  		the output of the sequential version

For the OpenMP parallel execution on multi-cores:

  make display_openmp : build first if needed, run if needed, then display
  		the output of the OpenMP parallel version

For the CUDA parallel execution on nVidia GPU:

  make display_cuda : build first if needed, run if needed, then display
  		the output of the CUDA parallel version

For the CUDA optimized and parallel execution on nVidia GPU:

  make display_cuda-opt : build first if needed, run if needed, then display
  		the output of the optimized CUDA parallel version

For an OpenMP parallel emulation of a GPU-like accelerator (useful for
debugging, without any GPU):

  make display_accel : build first if needed, run if needed, then display
  		the output of the accelerator OpenMP parallel emulation version

For the OpenCL parallel execution on nVidia GPU:

  make display_opencl : build first if needed, run if needed, then display
  		the output of the OpenCL parallel version.

