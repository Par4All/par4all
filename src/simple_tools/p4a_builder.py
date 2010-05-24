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

class p4a_build():
    '''Par4All builder class. For now everything is in the ctor. But the class should have methods some day.'''
    
    def __init__(self, files, output_file, 
        cppflags = [], cflags = [], ldflags = [], nvccflags = [],
        extra_obj = [], 
        cc = None, ld = None, ar = None, nvcc = None,
        debug = False, optimize = True, openmp = False, icc = False, arch = None):
        
        (base, ext) = os.path.splitext(output_file)
        
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
        
        prepend_cflags = [ "-fno-strict-aliasing", "-fPIC" ]
        #prepend_cflags += [ "-fPIE" ]
        if get_verbosity() >= 2:
            prepend_cflags += [ "-v" ]
        if get_verbosity() >= 1:
            prepend_cflags += [ "-Wall" ]
        if debug:
            prepend_cflags += [ "-g" ]
        elif optimize:
            if icc:
                prepend_cflags += [ "-fast" ]
            else:
                prepend_cflags += [ "-O2" ]
        if openmp:
            if icc:
                prepend_cflags += [ "-openmp" ]
            else:
                prepend_cflags += [ "-fopenmp" ]
        cflags = prepend_cflags + cflags
        #ldflags = [ "-pie" ] + ldflags
        
        compile_files = []
        obj_files = []
        final_files = []
        final_command = cc
        cxx = False
        cuda = False
        
        machine_arch = get_machine_arch()
        lib_arch_suffix = ""
        arch_flags = []
        if arch is None:
            if machine_arch == "x86_64":
                lib_arch_suffix = "_x86_64" # fix for cuda libs ..
        else:
            if arch == 32 or arch == "32" or arch == "i386":
                arch_flags = [ "-m32" ]
            elif arch == 64 or arch == "64" or arch == "x86_64":
                arch_flags = [ "-m64" ]
                lib_arch_suffix = "_x86_64"
            else:
                raise p4a_error("Unsupported architecture: " + arch)
        
        cuda_cppflags = []
        cuda_ldflags = []
        nvidia_sdk_dir = os.path.expanduser("~/NVIDIA_GPU_Computing_SDK")
        if "NVIDIA_SDK_DIR" in os.environ:
            nvidia_sdk_dir = os.environ["NVIDIA_SDK_DIR"]
        
        test_cuda_include_dirs = [ "/usr/local/cuda/include", os.path.join(nvidia_sdk_dir, "C/common/inc") ]
        if "CUDA_INCLUDE_DIRS" in os.environ:
            test_cuda_include_dirs += os.path.expanduser(os.environ["CUDA_INCLUDE_DIRS"].split(":"))
        for dir in test_cuda_include_dirs:
            if os.path.isdir(dir):
                cuda_cppflags += [ "-I" + dir ]
        
        #main_cuda_lib_dir = None
        #if lib_arch_suffix == "x86_64":
        #    main_cuda_lib_dir = "/usr/local/cuda/lib64"
        #else:
        #    main_cuda_lib_dir = "/usr/local/cuda/lib"
        # It should be OK to -L both 64bit and 32bit version because the compiler will pick the right libs
        # for the requested architecture.
        test_cuda_lib_dirs = [ "/usr/local/cuda/lib64", "/usr/local/cuda/lib", 
            os.path.join(nvidia_sdk_dir, "C/lib"), 
            os.path.join(nvidia_sdk_dir, "C/common/lib/linux") ]
        if "CUDA_LIB_DIRS" in os.environ:
            test_cuda_lib_dirs += os.path.expanduser(os.environ["CUDA_LIB_DIRS"].split(":"))
        for dir in test_cuda_lib_dirs:
            if os.path.isdir(dir):
                cuda_ldflags += [ "-L" + dir ]
        cuda_ldflags += [ "-Bdynamic -lcudart" ] #, "-Bstatic -lcutil" + lib_arch_suffix ]
        
        for file in files:
            (b, e) = os.path.splitext(file)
            if e == ".cu":
                run2([ nvcc, "--cuda" ] + cppflags + cuda_cppflags + nvccflags + [ file ])
                compile_files += [ file + ".cpp" ]
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
        
        if cuda:
            if ext == ".a":
                raise p4a_error("Cannot build a shared library when using cuda")
        
        if cuda and ext == "" and icc:
            pass
            # Compiling CUDA enabled executable with ICC is tricky ...
            #ldflags = [ "-pie" ] + ldflags #"-Bdynamic", 
            #cflags = [ "-fPIE", "-static" ] + cflags
        
        for file in compile_files:
            obj_file = change_file_ext(file, ".o")
            run2([ cc, "-c" ] + cppflags + arch_flags + cflags + [ "-o", obj_file, file ])
            obj_files += [ obj_file ]
        
        if ext == ".o":
            if len(obj_files) == 1:
                shutil.move(obj_files[0], output_file)
            return
        
        final_files += obj_files
        final_files += extra_obj
        
        prefix_flags = []
        if ext == ".a":
            prefix_flags += [ "-static" ]
        elif ext == ".so":
            prefix_flags += [ "-shared" ]
        elif ext == "":
            pass
        else:
            raise p4a_error("Unsupported extension for output file: " + output_file)
        
        if cuda:
            cppflags += cuda_cppflags
            ldflags += cuda_ldflags
        
        run2([ final_command ] + prefix_flags + ldflags + [ "-o", output_file ] + final_files)
        #if cxx:
        #ldflags += [ "-L`gcc -print-file-name=` /usr/lib/crt1.o /usr/lib/crti.o " ] #"-lstdc++" ]
        #run(" ".join([ final_command ] + prefix_flags + ldflags + [ "-o", output_file ] + final_files + ["/usr/lib/crtn.o -limf -lsvml -lm -lipgo -ldecimal -lgcc -lgcc_eh -lirc -lc -lgcc -lgcc_eh -lirc_s -ldl -lc"]))
        

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
