Here are given small examples showing how to use p4a, tpips and PyPS to
parallelize some code.

People only interested in quick results should look only at examples/P4A
using p4a.

- P4A directory contains basic examples that use p4a directly to generate
  OpenMP, CUDA, optimized CUDA and OpenCL parallel programs:

  - Hyantes : a geographic application on potential smoothing;

  - Stars-PM : a particle-mesh N-body cosmological simulation, written by
    Dominique Aubert and Mehdi Amini;

  - Jacobi : example of simple Jacobi solver applied to an image;

  - saxpy_c99 : SAXPY (Single-precision real Alpha X Plus Y) combines
	scalar multiplication and vector addition, written in C99;

- the Benchmarks directory contains open source benchmarks that are used
  to measure Par4All performance, for example to publish some
  articles. Useful to reproduce article results too;

- SCMP contains an example from the SCALOPES ARTEMIS European project
  targeting the SCMP data-flow architecture from CEA;

- F77_matmul_OpenMP shows how to use the PIPS CLI tpips for advanced user
  but also simple p4a for Fortran;

- Python contains examples for bleeding edge users that want to tailor
  Par4All to their own needs by dynamically changing the behaviour with
  some Python injections.
