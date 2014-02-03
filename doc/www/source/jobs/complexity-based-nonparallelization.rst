Complexity-based (non)parallelization
=====================================

Master or engineer internship

**Keywords**: compilation, parallelization, complexity analysis, program
transformations

Manycore processors and heterogeneous accelerators allow good performance
on parallel programs with good power efficiency but their programming is
quite challenging.

There exist some frameworks to ease application development on these
platforms and Par4All (http://par4all.org) is one of them, based on
automatic parallelization that can parallelize C and Fortran sequential
programs to OpenMP, CUDA and OpenCL.

Unfortunately, sometimes the automatic parallelization is successful on
some parts where it is counterproductive to have a parallel execution,
because there is not enough parallelism to compensate the start-up
overhead, or there is too many data to transfer compared with the
computing intensity. In this case it is interesting not to parallelize
according to some cost function.

This internship deals with extending Par4All with some heuristics based on
a complexity analysis relying on polyhedral model and Ehrhart polynomial
(http://en.wikipedia.org/wiki/Ehrhart_polynomial) already in PIPS
(http://pips4u.org) but to be extended to deal with the C language. With
it, some estimations of the computational intensity and the CPU-GPU
communication costs will be computed.

In some cases it is also interesting to do some parallel promotion, when a
sequential part is executed redundantly in parallel to avoid having a
sequence of parallel/sequential/parallel parts. This approach is also to
be studied with the complexity analysis.

This internship can be followed by a job or a PhD thesis.

Some knowledge useful for this project: compilation, C, Python, Unix.

Advisor : Thierry.porcher  at hpc-project dot com,  http://par4all.org

HPC Project http://www.hpc-project.com is a start-up with around 35 people.

Meudon (92) or Montpellier (34), France.

..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "american"
  ### End:
