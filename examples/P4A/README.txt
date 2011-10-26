If you do not have write access to this directory, copy its content (with
cp -a for example) somewhere you can write into and try the examples from
this new location.

This directory contains par4all basic examples : Hyantes, Jacobi, Stars-pm 
and saxpy_c99. To run one example of this directory, for example to run Hyantes :

$ cd Hyantes	

Then you can parallelize, build and execute that example using the following 
commands. These commands are generic, and more specific commands can be 
found in the directory of each example.

For the sequential execution

  make seq : build the sequential program

  make run_seq : build first if needed, then run the sequential program

For the OpenMP parallel execution on multi-cores:

  make openmp : parallelize the code to OpenMP source and compile

  make run_openmp : build first if needed, then run the OpenMP parallel
  program

For the CUDA parallel execution on nVidia GPU:

  make cuda : parallelize the code to CUDA source and compile

  make run_cuda : build first if needed, then run the CUDA parallel
  program

    Do not forget to have the CUDA runtime correctly
    installed. LD_LIBRARY_PATH should contain at least the location of
    CUDA runtime library.

For the CUDA optimized and parallel execution on nVidia GPU:

  make cuda-opt : parallelize the code to CUDA source and compile

  make run_cuda-opt : build first if needed, then run the CUDA parallel
  program

For an OpenMP parallel emulation of a GPU-like accelerator (useful for
debugging, without any GPU):

  make accel : parallelize the code to GPU-like OpenMP source and compile

  make run_accel : build first if needed, then run the parallel program

For the OpenCL parallel execution on nVidia GPU:

  make opencl : parallelize the code to OpenCL sources: the host program sources
  *.p4a.c and the kernel program sources *.cl

  make run_opencl : build first if needed, then run the OpenCL parallel
  program

	To compile for nVidia GPU, you need CUDA environment installed. 

To chain all the demos:

  make demo

You might also run "make clean" first to force rebuilding.

