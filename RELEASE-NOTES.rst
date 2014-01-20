Release notes for Par4All 1.4.2, 2012/06/22
===========================================

Par4All is an open-source environment to do source-to-source
transformations on C and Fortran programs for parallelizing, optimizing,
instrumenting, reverse-engineering, etc. on various targets, from embedded
multiprocessor system-on-chip with hardware accelerators up to high
performance computer and GPU.


Introduction
------------

Par4All parallelizes C programs to
OpenMP, CUDA and OpenCL, and Fortran programs to OpenMP.

This release presents new features of Par4All and PIPS that are described
deeper in the ``changelog`` file. For all features and changes in the
previous versions, please see the release notes archives from git.

Par4All is provided as Debian and Ubuntu Linux packages to ease
installation but can be also compiled on Fedora Linux. More detailed
information on installation can be found in the Par4All Installation
guide.

More documentation and information is available on http://par4all.org.

If you need more complex transformations, parallelizations and
optimizations, you may contact SILKAN for professional support and
you can mail to: support@par4all.org


Additions
---------

  - OpenCL:

    This version of Par4All generates kernel codes to be executed on GPU. To
    compile and to run these codes, CUDA environment should have been
    installed. For example you can transform a C program example.c to kernel
    codes in ``*.cl`` files and host codes in ``*.p4a.c`` files, and then
    compile the host codes to obtain an executable::

      p4a --opencl example.c -o example

  - C99 option for CUDA code generation:

    Par4All can deal with C99 code sources. Indeed ``nvcc`` doesn't support the
    following C99 syntax::

        f_type my_function(size_t n, size_t m,
                           float array[n][m]) {
            ...
        }

    Using ``--c99`` option, ``p4a`` will automatically generate the CUDA code
    in new C89 files (with no VLA but pointers with linearized accesses
    instead) that will be compiled by ``nvcc``. A simple call to each kernel
    will be inserted into the original files ``*.p4a.c`` that can be compiled
    with your usual C99 compiler, and linked to the object files compiled from
    ``*.cu``, to produce an executable.

    Example::

      p4a --cuda --c99 c99_example.c -o c99_example

  - SCMP:

    Generation of codes for the SCMP (Scalable Chip MultiProcessor) data-flow
    architecture from CEA, funded by SCALOPES ARTEMIS European project. For
    example the following command will generate an application for the SCMP
    architecture from a C program::

      p4a --scmp example.c

  - Easier management of compute capabilities of CUDA target, using
    ``--cuda-cc`` option

  - Atomic operations for parallelizing reductions on GPU

  - Can use PoCC to optimize some loop nests


Knowing issues
--------------

- C:

  - you should avoid heavy use of pointers to enable a good
    parallelization. The point-to analysis is still on-going in PIPS, but
    in the general case it is untractable... So avoid if possible. Use
    nice C99 multidimensional arrays with dynamic size (as in Fortran for
    long time) and pointer to arrays instead. This is documented into the
    coding rules, found at http://download.par4all.org/doc/p4a_coding_rules

  - there are still issues in the part that analyzes the control graph
    (the PIPS ``controlizer``). For example, right now we cannot deal well
    with a different hierarchy between the control graph and the variable
    scoping. For example if you have a block with a variable declaration
    within and a ``goto`` to/from outside, it may probably fail. This
    happens also if you have ``break`` or ``continue`` in loops with some
    local variables since those are internally represented with ``goto``.

- CUDA & OpenCL:

  - In this release the shared memory is not used yet, so to get the best
    performance, loop nests need to be quite compute intensive, typically with
    regular accesses to memory. Hopefully, the new cache architecture in Fermi
    GPU is interesting to balance this limitation.

  - Par4All does not generate additional tiling yet, so right now the
    iteration spaces of the 3 outer loops of a parallel loop nest are limited
    by hardware GPU limits.




This text is typeset according to the reStructuredText Markup
Specification. For more information:
http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html

This matters because this file is processed to build higher order
documentation on Par4All.

%%% Local Variables:
%%% mode: rst
%%% ispell-local-dictionary: "american"
%%% End:
