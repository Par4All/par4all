If you do not have write access to this directory, copy its content (with
cp -a for example) somewhere you can write into and try the examples from
this new location.

# To chain all the demos:
make demo



Jacobi is a simple relaxation on an image that is done T times (400 times in
this demo). The convolution kernel is the matrix
0   1/4  0
1/4  0  1/4
0   1/4  0

It uses the top left part of the HPC Project logo as input

# To chain all the Jacobi demos:
make demo

For the sequential execution

  make seq : build the sequential program

  make run_seq : build first if needed, then run the sequential program

  make display_seq : build first if needed, run if needed, then display
  the output of the sequential version

For the OpenMP parallel execution on multicores:

  make openmp : parallelize the code to OpenMP source and compile

  make run_openmp : build first if needed, then run the OpenMP parallel
  program

  make display_openmp : build first if needed, run if needed, then display
  the output of the OpenMP parallel version

For the CUDA parallel execution on nVidia GPU:

  make cuda : parallelize the code to CUDA source and compile

  make run_cuda : build first if needed, then run the CUDA parallel
  program

    Do not forget to have the CUDA runtime correctly
    installed. LD_LIBRARY_PATH should contain at least the location of
    CUDA runtime library.

  make display_cuda : build first if needed, run if needed, then display
  the output of the CUDA parallel version

For the CUDA parallel execution on nVidia GPU with communication optimization:

  make cuda_opt : parallelize the code to CUDA source and compile

  make run_cuda_opt : build first if needed, then run the CUDA parallel
  program

    Do not forget to have the CUDA runtime correctly
    installed. LD_LIBRARY_PATH should contain at least the location of
    CUDA runtime library.

  make display_cuda_opt : build first if needed, run if needed, then display
  the output of the CUDA parallel version

For an OpenMP parallel emulation of a GPU-like accelerator (useful for
debugging, without any GPU):

  make accel : parallelize the code to GPU-like OpenMP source and compile

  make run_accel : build first if needed, then run the parallel program

  make display_accel : build first if needed, run if needed, then display
  the output of the accelerator OpenMP parallel emulation version

For an OpenMP parallel emulation of a GPU-like accelerator (useful for
debugging, without any GPU) with communication optimization:

  make accel_opt : parallelize the code to GPU-like OpenMP source and compile

  make run_accel_opt : build first if needed, then run the parallel program

  make display_accel_opt : build first if needed, run if needed, then display
  the output of the accelerator OpenMP parallel emulation version
