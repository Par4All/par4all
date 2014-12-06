Features
========

Par4All is an automatic parallelizing and optimizing compiler (workbench)
for C and Fortran sequential programs.

The purpose of this source-to-source compiler is to adapt existing
applications to various hardware targets such as multicore systems, high
performance computers and GPUs. It creates a new source code and thus
allows the original source code of the application to remain unchanged.

The main benefits of this source-to-source approach are:

- to keep the original source code of the application free from
  modifications,

- to obtain generated parallelized sources for various hardware platforms,

- to rely on vendor’s optimized target tools,

- to be able to optimize manually the generated source code.

The generated sources just need to be processed through the usual compilers:

-  optimized compilers for a given processor,

- vendor compilers for embedded processors,

- CUDA,

- OpenCL,

- OpenMP,

- linkable with MPI and other libraries.


Par4All current version
-----------------------

The current 1.*x* version can generate CUDA and OpenCL code from C code
and OpenMP from C and Fortran 77 code with a simple easy-to-use high-level
script ``p4a``. With this script, you can get a parallelized version of
your sources or even call the backend compiler to generate executable
binaries with ``gcc``, ``nvcc`` or ``icc`` for example.

On the `benchmarks <benchmarks>`_ page, there are some performance results
with Par4All on multicores and GPU.

Currently there is no support for Windows. Mac OS X may work by compiling
from the sources but is not supported. But you can use a virtual machine
with Ubuntu or Debian 64-bit *x*\ 86 on these systems to generate parallel
versions of your programs.


What is going on?
-----------------

The main development of the 1.4 branch is almost stopped since we are
focusing our developments on the 2.0 version based on Clang that takes
most of our time.

**Warning:** since the project is no longer supported by SILKAN, most of
the developments are frozen, such as the Clang/LLVM/SoSlang for 2.x
version. :-(

- Switching to Clang as the base framework for Par4All 2.*x*

- Scilab/Xcos and MATLAB/Simulink to OpenMP/CUDA/OpenCL with Wild Cruncher

- Python compilation & parallelization to OpenMP/CUDA/OpenCL

- Code generation for more embedded systems (Tilera, Kalray MPPA, ST
  P2012/STORM)

- More user-friendly interfaces (Eclipse...)

- Improving vector code generation (*x*\ 86 SSE & AVX, ARM Neon, CUDA and
  OpenCL vectors)

- Better CUDA and OpenCL generation (loop fusion, shared memory...)

- Improving the OpenMP output

- Automatic instrumentation for loop parameters extraction at runtime

- Java compilation & parallelization to OpenMP/CUDA/OpenCL

- Finish the Fortran 95 support with the ``gcc``/``gfc`` front-end


Roadmap
-------

- Par4All 0.1 and 0.2 went out to provide Fortran 77 to OpenMP
  parallelization to modernize legacy code and C to OpenMP
  parallelization. There were first releases to test the integration
  process and were not really distributed as packages or with high level
  compilation scripts;

- Par4All 1.0 (07/2010) parallelizes Fortran and C to OpenMP and C to CUDA
  and is the first easy-to-use public version;

- Par4All 1.1 (03/2011) deals with C99 and introduces basic support for
  Fortran 95 to OpenMP;

- Par4All 1.2 (07/2011) loop-fusion and communication optimizations for
  CUDA;

- Par4All 1.3.1 (01/2012) generates OpenCL;

- Par4All 1.4.3 (09/2012) deal with Spear-DE output;

- Par4All 2.0 : new version based on Clang/LLVM. The developments are on
  hold...


Internals
---------

Internally, Par4All is currently composed of different components:

- the `PIPS <http://pips4u.org>`_ source-to-source compiler that began at
  `MINES ParisTech <http://cri.mines-paristech.fr>`_ in 1988 and is
  currently developed also in many other places: `SILKAN
  <http://www.silkan.com>`_, `Institut TÉLÉCOM/TÉLÉCOM Bretagne
  <http://departements.telecom-bretagne.eu/info>`_, `IT SudParis
  <http://inf.telecom-sudparis.eu>`_, `RPI (Rensselaer Polytechnic
  Institute) <http://www.cs.rpi.edu>`_.

- the `PolyLib <http://icps.u-strasbg.fr/polylib/>`_ used by PIPS,

- GCC/GFC for the Fortran95 parser,

- and of course own tools and scripts to make all these components and the
  global infrastructure usable.

Par4All is an open source project that merges various open source
developments. `More info on the community here <community>`_.

..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: rst,flyspell
  ### ispell-local-dictionary: "american"
  ### End:
