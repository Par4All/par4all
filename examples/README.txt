Here are given small examples showing how to use p4a, tpips and pyps to
parallelize some code.

People only interested in quick results should look only at examples using p4a.

- P4A directory contains basic examples that use p4a directly to generate
  OpenMP or CUDA parallel programs:

  - Hyantes : a geographic application on potential smoothening

- F77_matmul_OpenMP shows how to use the PIPS CLI tpips for advanced user
  but also simple p4a for Fortran

- P4A_accel/Jacobi is an example for advanced users wanting to understand
  how it works in detail. It shows an example to perform parallelization
  and optimization on a Jacobi iterative solver. It uses pips_c2openmp
  simple script for OpenMP and a specific pyps script for CUDA generation.

- Python contains examples for bleeding edge users that want to tailor
  Par4All to their own needs by dynamically changing the behaviour with
  some Python injections.
