#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Ronan Keryell <ronan.keryell@hpc-project.com>
#

'''
Par4All stubs infrastructure

PIPS use interprocedural analysis to transform and parallelize programs.
In this way a lot of details are exposed to PIPS so that it can trace
memory access, arrays... and being able to know if some parts of the
program can be parallel or not.

But there are some case where you cannot provide the sources: the system
input/outputs (read(), write()...) various standard libraries (printf(),
rand(), malloc()...) or already finely hand-tuned parallel (FFTW,
CuBLAS...) code or whatever we do not have the source (MPI...).

In this case you can not use PIPS. :-( Of course there is a work around. :-)

For C standard stuff, a memory-effect equivalent of the intrinsic
functions is defined directly in PIPS (in bootstrap) so that it should
work.

But in other cases, you have to provide some source code for the functions
you want to use in an opaque way that mimic the memory effect on the
parameter. This is called the stub functions. In this way PIPS can have a
vague idea of what the function will do and if it can parallelize code
that uses this function.

Of course to produce a real program at the end, we have to provide some
source code for the real implementation to be compiled or an already
compiled library to link with.

The aim of this module is specifically to help defining in Par4All the
stubs and implementations for some functions.

'''

#import sys, os, re, shutil
import p4a_util

# All the stubs defined in Par4All indexed by their name:
existing_stubs = {}


def define(**args):
    """
    Method to define a new stub set

    @param name is mandatory and refers to this stub for example on the
    options given to Par4All. It is a string or a function returning a
    string.

    @param stub_dir specifies where are located the file used as stub. It is
    a string or a function returning a string. For example
    p4a_util.get_safe_environ(environment_variable_name) to get the value
    of environment_variable_name or display an error message if it is not
    defined.

    @param stub_c defines the C source file names defining some stubs. It
    is a string, a string list or a function returning them.

    @param stub_f defines the Fortran source file names defining some
    stubs. It is a string, a string list or a function returning them

    @param implementation_dir specifies where are located the
    implementation source files or library. It is a string or a function
    returning a string.

    @param implementation_c defines the implementation C source files to
    compile. It is a string, a string list or a function returning them.

    @param implementation_f defines the implementation Fortran source
    files to compile. It is a string, a string list or a function
    returning them.

    @param implementation_c_lib defines the C library implementation names
    to link with -l. It is a string, a string list or a function returning
    them.
    """

    # Get the mandatory name and create 
    stub = Stub(p4a_util.get_arg_by_name(args, "name"))
    p4a_utils.set_attribute_as_string_if_defined(stub, args, "stub_dir")
    p4a_utils.set_attribute_as_strings_if_defined(stub, args, "stub_c")
# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
