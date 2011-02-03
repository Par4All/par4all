Here are given small examples showing how to use p4a, tpips and PyPS to
parallelize some code.

People only interested in quick results should look only at examples using p4a.

- P4A directory contains basic examples that use p4a directly to generate
  OpenMP or CUDA parallel programs:

  - Hyantes : a geographic application on potential smoothening

  - Stars-pm : a particle-mesh N-body cosmological simulation, written by
    Dominique Aubert and Mehdi Amini.

  - Jacobi : example of simple Jacobi solver applied to an image


- F77_matmul_OpenMP shows how to use the PIPS CLI tpips for advanced user
  but also simple p4a for Fortran

- Python contains examples for bleeding edge users that want to tailor
  Par4All to their own needs by dynamically changing the behaviour with
  some Python injections.
