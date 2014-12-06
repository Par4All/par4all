Par4All version 1.4.5
=====================

The WWW site for the project is http://www.par4all.org

Par4all merges various free and open source developments. Par4All aims to
achieve the migration of software to multi-core and other parallel
processors, as well as to accelerating processors such as GPU. It is based
on multiple components, including the source-to-source compiler framework
`PIPS <http://pips4u.org>`_ (Inter Procedural Parallelization of
scientific programs, the oldest research compiler still alive on Earth
\o/), and is developed by `SILKAN <http://www.silkan.com>`_, `MINES
ParisTech <http://cri.mines-paristech.fr>`_ and `Institut Télécom
<http://departements.telecom-bretagne.eu/info/>`_.

With one command line, Par4All automatically transforms C and Fortran
sequential programs to parallel ones. It will offer code execution
optimization on multi-core and many-core architectures without using any
specific programming language.

Current version of Par4All takes C programs as input and generates OpenMP,
CUDA and OpenCL programs. It also can transform Fortran programs to
OpenMP. Further, these generated files can be compiled to get an
executable to be executed on the target platforms, such as multi-core and
GPU.

The main user interface in Par4All is the command-line interface ``p4a``
to invoke parallelization of the provided source codes, but also back-end
compilation or automatic ``CMakeFile`` generation.

But the other commands from the PIPS project included in Par4All are also
available: ``tpips`` and ``ipyps``. Of course, they are reserved for quite
more advanced users. For more information, look at http://pips4u.org

Since ``p4a`` is a script that interacts with PIPS to automatically
parallelize the source code in an average way, it is of course interesting
to dig into PIPS to apply specific transformations or change the value of
some parameters to get better performance on a given application.

More on Par4All features:

- Par4All supports almost complete C99;

- Fortran 77 with some extensions is to be used for production. Work on
  Fortran95 is still in progress and is far from being usable yet;

- OpenCL & CUDA:

  Par4All uses a static dataflow analysis to optimize communications
  between host and GPU and to remove redundant GPU array allocation. Have
  a look at ``--com-optimization`` option;

- embedded systems:

  with OpenMP output, as any other shared memory multiprocessor systems,
  all the embedded systems that accept OpenMP can be addressed.  For
  example in the SCALOPES ARTEMIS European project, the code generation
  for the Scaleo Chip Leon 3 MP-SoC is done through the OpenMP support of
  the target.

New features :

- This version of Par4All can generate OpenCL host and kernel codes.
- Generation codes for the SCMP dataflow architecture from CEA.
- Generation for SIMILAN project.
- Some more options on ``p4a``, such as ``--atomic`` to use atomic operations
  for parallelizing reductions on GPU, ``--com-optimization`` to optimize
  communications between the host and the GPU, ``--kernel-unroll=...`` to
  unroll loops inside kernels...

Look at the `changelog <src/simple_tools/DEBIAN/changelog>`_ file for more
details.


Hardware and software requirements
----------------------------------

To install Par4All, GNU/Linux Debian or Ubuntu is preferred. For more
detailed information about Par4All requirements and installation, please
refer to Par4All installation guide.


Contact
-------

**Warning**: Since the project is no longer supported by SILKAN, the
  following is for information only. Use GitHub mechanisms instead.

| SILKAN
| 9, route du Colonel Marcel Moraine
| 92360 Meudon La Forêt
| FRANCE
| Phone: +33 1 46 01 03 27
| Fax: +33 1 46 01 05 46

Par4All support : support at par4all.org

11/11/11, 11:11:11


..
  # Some Emacs stuff:
  ### Local Variables:
  ### mode: flyspell
  ### ispell-local-dictionary: "american"
  ### End:
