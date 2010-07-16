# To chain all the demos:
make demo



Jacobi is a simple relaxation on an image that is done T times (400 times in
this demo). The convolution kernel is the matrix
0   1/4  0
1/4  0  1/4
0   1/4  0

It uses the top left part of the HPC Project logo as input

# To chain all the Jacobi demos:
make demo_jacobi

# To compile and run the sequential version:
make jacobi
make run_jacobi (note that it compiles too if needed...)
The reference source is in jacobi.c

# To compile and run the Par4All OpenMP version:
make jacobi_omp
make run_jacobi_omp (note that is compile too if needed...)
The generated source is in jacobi_omp.c

# To compile and run with the Par4All P4A_accel runtime:

- CUDA version:
make jacobi_p4a_cuda
make run_jacobi_p4a_cuda (note that it compiles too if needed...)

There is not a big improvement compared with sequential or OpenMP version
because we do not used shared memory yet, but the CPU can take advantage of
its cache.

- Par4All P4A_accel runtime OpenMP version (with useless memory copies):
make jacobi_p4a_omp
make run_jacobi_p4a_omp (note that it compiles too if needed...)
The generated source files are in jacobi.database/P4A
