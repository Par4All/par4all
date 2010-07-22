If you do not have write access to this directory, copy its content (with
cp -a for example) somewhere you can write into and try the examples from
this new location.

This is a simple example to demonstrate what pips can achieve on Fortran77
programs with 2 different methods by using an explicit .tpips script or
directly with p4a. Here you will automatically parallelize a matrix
multiplication program using the PIPS environment in Par4All.

In this directory you should find the following files:
   - Makefile     : A simple makefile to build the parallel and the sequential
            version of the matrix multiplication program (see bellow
            for details).
   - matmul.f     : The matrix multiplication source code
   - matmul.tpips : The tpips script that parallelizes the matmul.f program.
     It is a minimal program you can start from to build more complex scripts
   - PipsCheck.sh : A simple script to test if the PIPS environment is set up
   - README   : This file you are reading

1 - Do it fast

    1 Set up your Par4All environment (www.par4all.org for details if not
    done yet)

    2 You may need to install the lib gomp (OpenMP runtime) and you also
      need to change the limit of the stack size in your shell using the
      following command: "ulimit -s unlimited"

    3 Run the "make" command -> this builds two executable versions from
      the same source file, one sequential and one parallel.

    4 execute "time ./matmul"

    5 execute "time ./matmul_par"

      If you get a segmentation violation, it is probably because your system
      does not allow enough memory on the stack to allocate the matrices.

      Have you tried a "ulimit -s unlimited" or equivalent according to
      your shell?

    6 Great you have already produced and executed two program versions from the
      same source file. Easy, isn't it?

    7 Now you can evaluate the speed up on your machine using this simple
      formula : sequential_elapse_time / parallel_sequential_time. As an
      example on a workstation with 2 Intel Xeon X5440 the score of 6.6 is
      reached, with 2 Intel Xeon X5670 @ 2.93GHz (12 cores) we have a
      speed-up of 11.6

    8 The directory can be clean using the "make clean" command

2 - The Makefile

Execute "make" to build both the parallel version and the sequential
version of the program.

The "make parallel" and "make sequential" commands build respectively the
parallel and sequential version of the program. Note that the building process
of the parallel program also include the source file generation by PIPS.

3 - The Matrix Multiplication

The Matrix multiplication basically allocates three square matrices (with
2000 rows). Then it initializes the matrices with 1's and do the
multiplication. It finally checks the result matrix.

4 - The tpips script

This tpips script is used to instruct tpips the transformations to apply
on the matmul.f source code. Here 5 phases are applied to get an OpenMP
version of the source code.

Once the PIPS environment is loaded, the "tpips matmul.tpips" command will
produce the parallel version of matmul.f in the directory matmul.data/Src.

You can see that all the loops are parallelized, internal loop indices are
privatized and reductions are detected, even on boolean values.

5 - Make it even easily with p4a (so without any .tpips to write):

  p4a matmul.f -o matmul_p4a

  time ./matmul_p4a

  At HPC Project, we get a speed-up of 12.1 with 2 Intel Xeon X5670 @
  2.93GHz (12 cores)

Any question and remarks : support@par4all.org
