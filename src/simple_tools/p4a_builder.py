#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Par4All Builder Class
'''

import sys, os, re, shutil
from p4a_util import *

def make_safe_intermediate_file_path(input_file, build_dir, change_ext = None):
    '''Make up a safe intermediate file path.'''
    intermediate_file = ""
    while True:
        intermediate_file = file_add_suffix(os.path.join(build_dir, os.path.split(input_file)[1]), "_" + gen_name(prefix = ""))
        if change_ext:
            intermediate_file = change_file_ext(intermediate_file, change_ext)
        if not os.path.exists(intermediate_file):
            break
    return intermediate_file

class p4a_builder():
    '''Par4All builder class. For now everything basically is in the constructor. But this class should have methods some day.'''
    
    def __init__(self,
        files,
        output_file, 
        cppflags = [],
        cflags = [],
        ldflags = [],
        nvccflags = [],
        extra_obj = [], 
        cc = None, 
        ld = None, 
        ar = None, 
        nvcc = None,
        debug_flags = False, 
        optimize = True, 
        openmp = False, 
        icc = False, 
        arch = None,
        build_dir = None
        ):
        
        # Determine build directory.
        if not build_dir:
            build_dir = os.path.join(os.getcwd(), ".build")
        debug("Build dir: " + build_dir)
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        
        # Get output file extension. 
        # This will be used later to determine compilation behaviour.
        ext = get_file_ext(output_file)
        
        # Set compiler defaults.
        if not nvcc:
            nvcc = "nvcc"
        if icc:
            if not which("icc"):
                raise p4a_error("icc is not available -- have you source'd iccvars.sh or iccvars_intel64.sh yet?")
            cc = "icc"
            ld = "xild"
            ar = "xiar"
        else:
            if not cc:
                cc = "gcc"
            if not ld:
                ld = "ld"
            if not ar:
                ar = "ar"
        
        # Make up the default C flags:
        prepend_cflags = [ "-fno-strict-aliasing", "-fPIC" ]
        #if get_verbosity() >= 2:
        #    prepend_cflags += [ "-v" ]
        if get_verbosity() >= 1:
            prepend_cflags += [ "-Wall" ]
        if debug_flags:
            prepend_cflags += [ "-g" ]
        elif optimize:
            if icc:
                prepend_cflags += [ "-fast" ]
            else:
                prepend_cflags += [ "-O2" ]
        if openmp:
            if icc:
                prepend_cflags += [ "-openmp" ]
                ldflags += [ "-openmp" ]
            else:
                prepend_cflags += [ "-fopenmp" ]
                ldflags += [ "-fopenmp" ]
        # Prepend our default C flags to the passed C flags:
        cflags = prepend_cflags + cflags
        
        # Initialize some variables.
        compile_files = []
        obj_files = []
        final_files = []
        final_command = cc
        cxx = False
        cuda = False
        
        machine_arch = get_machine_arch()
        arch_flags = []
        m64 = False
        if arch is None:
            if machine_arch == "x86_64":
                m64 = True
        else:
            if arch == 32 or arch == "32" or arch == "i386":
                arch_flags = [ "-m32" ]
            elif arch == 64 or arch == "64" or arch == "x86_64" or arch == "amd64":
                arch_flags = [ "-m64" ]
                m64 = True
            else:
                raise p4a_error("Unsupported architecture: " + arch)
        
        # Preprocess file list:
        for file in files:
            e = get_file_ext(file)
            if e == ".cu":
                # Run CUDA on .cu files to produce a .cpp.
                cuda_output_file = make_safe_intermediate_file_path(file, build_dir, change_ext = ".cu.cpp")
                run2([ nvcc, "--cuda" ] + cppflags + nvccflags + [ "-o", cuda_output_file, file ])
                compile_files += [ cuda_output_file ]
                cuda = True
                cxx = True
            elif e == ".c":
                compile_files += [ file ]
            elif e == ".cpp" or e == ".cxx":
                compile_files += [ file ]
                cxx = True
            elif e == ".f":
                final_files += [ file ]
                final_command = "gfortran"
            else:
                raise p4a_error("Unsupported extension for input file: " + file)
        
        # Option checking.
        if cuda and ext == ".a":
            raise p4a_error("Cannot build a shared library when using cuda")
        
        # Compile source files as object (.o) files.
        for file in compile_files:
            obj_file = make_safe_intermediate_file_path(file, build_dir, change_ext = ".o")
            run2([ cc, "-c" ] + cppflags + arch_flags + cflags + [ "-o", obj_file, file ])
            obj_files += [ obj_file ]
        
        # If we were requested to build a .o file... Well check if there are several of them
        # and warn the user if it is the case.
        if ext == ".o":
            if len(obj_files) >= 1:
                shutil.move(obj_files[0], output_file)
                if len(obj_files) > 1:
                    warn("You requested to compile " + ", ".join(compile_files) + " as an object (.o) file, but several have been generated in " + build_dir)
                return
            else:
                raise p4a_error("No object file generated!?")
        
        final_files += obj_files
        final_files += extra_obj
        
        # Add necesseray flags depending on the requested output file type.
        prefix_flags = []
        if ext == ".a":
            prefix_flags += [ "-static" ]
        elif ext == ".so":
            prefix_flags += [ "-shared" ]
        elif ext == "":
            pass
        else:
            raise p4a_error("Unsupported extension for output file: " + output_file)
        
        # Add necessary libraries for CUDA.
        if cuda:
            ldflags += get_cuda_ldflags(m64)
        
        # Run the final compilation step: produce the expected binary file.
        run2([ final_command ] + prefix_flags + ldflags + [ "-o", output_file ] + final_files)
        
        if os.path.exists(output_file):
            done("Generated " + output_file)
        else:
            warn("Expected output file " + output_file + " not found!?")


if __name__ == "__main__":
    print(__doc__)
    print("This module is not directly executable")

# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
