#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Par4All Builder Class
'''

import sys, os, re, shutil, time
from p4a_util import *

actual_script = change_file_ext(os.path.abspath(os.path.realpath(os.path.expanduser(__file__))), ".py", if_ext = ".pyc")
script_dir = os.path.split(actual_script)[0]

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

cuda_dir = ""
def get_cuda_dir():
    global cuda_dir
    if not cuda_dir:
        if "CUDA_DIR" in os.environ:
            cuda_dir = os.path.expanduser(os.environ["CUDA_DIR"])
        else:
            cuda_dir = "/usr/local/cuda"
            warn("CUDA_DIR environment variable undefined. Using '" +
                cuda_dir + "' as default location for CUDA installation")
        if not os.path.isdir(cuda_dir):
            raise p4a_error("CUDA directory not found or invalid (" + cuda_dir 
                + "). Please set the CUDA_DIR environment variable correctly")
    return cuda_dir

cuda_sdk_dir = ""
def get_cuda_sdk_dir():
    global cuda_sdk_dir
    if not cuda_sdk_dir:
        if "CUDA_SDK_DIR" in os.environ:
            cuda_sdk_dir = os.path.expanduser(os.environ["CUDA_SDK_DIR"])
        else:
            cuda_sdk_dir = os.path.expanduser("~/NVIDIA_GPU_Computing_SDK")
            warn("CUDA_SDK_DIR environment variable undefined. Using '" +
                cuda_sdk_dir + "' as default location for CUDA installation")
        if not os.path.isdir(cuda_sdk_dir):
            raise p4a_error("CUDA SDK directory not found or invalid (" + cuda_sdk_dir 
                + "). Please set the CUDA_SDK_DIR environment variable correctly")
    return cuda_sdk_dir

def get_cuda_cppflags():
    return [
        "-I" + os.path.join(get_cuda_dir(), "include"),
        "-I" + os.path.join(get_cuda_sdk_dir(), "C/common/inc"),
    ]

def get_cuda_ldflags(m64 = True, cutil = True, cublas = False, cufft = False):
    cuda_dir = get_cuda_dir()
    cuda_sdk_dir = get_cuda_sdk_dir()
    lib_arch_suffix = ""
    if m64:
        lib_arch_suffix = "_x86_64"
    flags = [
        "-L" + os.path.join(cuda_dir, "lib64"),
        "-L" + os.path.join(cuda_dir, "lib"),
        "-L" + os.path.join(cuda_sdk_dir, "C/lib"),
        "-L" + os.path.join(cuda_sdk_dir, "C/common/lib/linux"),
        "-Bdynamic -lcudart"
    ]
    if cutil:
        flags += [ "-Bstatic -lcutil" + lib_arch_suffix ]
    if cublas:
        die("TODO")
    if cufft:
        die("TODO")
    return flags

class p4a_builder():
    cppflags = []
    cflags = []
    ldflags = []
    nvccflags = []
    fortranflags = []
    
    cpp = None
    cc = None
    cxx = None
    ld = None
    ar = None
    nvcc = None
    fortran = None
    
    m64 = False
    cudafied = False
    extra_files = []
    
    def cudafy_flags(self):
        if self.cudafied:
            return
        self.cppflags += get_cuda_cppflags()
        self.ldflags += get_cuda_ldflags(self.m64)
        
        self.cppflags += [ "-DP4A_ACCEL_CUDA", "-I" + os.environ["P4A_ACCEL_DIR"] ]
        self.extra_files += [ os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_accel.cu") ]

        self.cudafied = True
    
    def __init__(self,
        cppflags = [], cflags = [], ldflags = [], nvccflags = [], fortranflags = [],
        cpp = None, cc = None, cxx = None, ld = None, ar = None, nvcc = None, fortran = None,
        arch = None,
        openmp = False, accel_openmp = False, icc = False, cuda = False,
        add_optimization_flags = False, no_default_flags = False
    ):
        
        if not nvcc:
            nvcc = "nvcc"
        if icc:
            if not which("icc"):
                raise p4a_error("icc is not available -- have you source'd iccvars.sh or iccvars_intel64.sh yet?")
            cxx = cc = "icc"
            ld = "xild"
            ar = "xiar"
        else:
            if not cxx:
                cxx = "g++"
            if not cc:
                cc = "gcc"
            if not ld:
                ld = "ld"
            if not ar:
                ar = "ar"
        if not cpp:
            cpp = cc + " -E"
        if not fortran:
            fortran = "gfortran"
        
        if add_optimization_flags:
            if icc:
                cflags += [ "-fast" ]
            else:
                cflags += [ "-O2" ]
        
        if openmp:
            if icc:
                cflags += [ "-openmp" ]
                ldflags += [ "-openmp" ]
            else:
                cflags += [ "-fopenmp" ]
                ldflags += [ "-fopenmp" ]
            
            if accel_openmp:
                cppflags += [ "-DP4A_ACCEL_OPENMP", "-I" + os.environ["P4A_ACCEL_DIR"] ]
                self.extra_files += [ os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_accel.c") ]
                
        if not no_default_flags:
            cflags = [ "-g", "-Wall", "-fno-strict-aliasing", "-fPIC" ] + cflags
        
        m64 = False
        machine_arch = get_machine_arch()
        if arch is None:
            if machine_arch == "x86_64":
                m64 = True
        else:
            if arch == 32 or arch == "32" or arch == "i386":
                cflags += [ "-m32" ]
            elif arch == 64 or arch == "64" or arch == "x86_64" or arch == "amd64":
                cflags += [ "-m64" ]
                m64 = True
            else:
                raise p4a_error("Unsupported architecture: " + arch)
        
        self.cppflags = cppflags
        self.cflags = cflags
        self.ldflags = ldflags
        self.nvccflags = nvccflags
        self.fortranflags = fortranflags
        
        self.cpp = cpp
        self.cc = cc
        self.cxx = cxx
        self.ld = ld
        self.ar = ar
        self.nvcc = nvcc
        self.fortran = fortran
        
        self.m64 = m64
        if cuda:
            self.cudafy_flags()

    def cu2cpp(self, file, output_file):
        run2([ self.nvcc, "--cuda" ] + self.cppflags + self.nvccflags + [ "-o", output_file, file ],
            #extra_env = dict(CPP = self.cpp) # Necessary?
        )

    def c2o(self, file, output_file):
        run2([ self.cc, "-c" ] + self.cppflags + self.cflags + [ "-o", output_file, file ])
    
    def cpp2o(self, file, output_file):
        run2([ self.cxx, "-c" ] + self.cppflags + self.cflags + [ "-o", output_file, file ])

    def f2o(self, file, output_file):
        run2([ self.fortran, "-c" ] + self.cppflags + self.fortranflags + [ "-o", output_file, file ])

    def build(self, files, output_files, extra_obj = [], build_dir = None):
        
        files += self.extra_files
        
        has_cuda = False
        has_c = False
        has_cxx = False
        has_fortran = False
        
        # Determine build directory.
        if not build_dir:
            build_dir = os.path.join(os.getcwd(), ".build")
        debug("Build dir: " + build_dir)
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        
        # First pass: make .c, .cpp or .f files out of other extensions (.cu, ..):
        first_pass_files = []
        for file in files:
            if cuda_file_p(file):
                has_cuda = True
                self.cudafy_flags()
                cucpp_file = make_safe_intermediate_file_path(file, build_dir, change_ext = ".cu.cpp")
                self.cu2cpp(file, cucpp_file)
                first_pass_files += [ cucpp_file ]
            else:
                first_pass_files += [ file ]
        
        # Second pass: make object files out of source files.
        second_pass_files = []
        for file in first_pass_files:
            if c_file_p(file):
                has_c = True
                obj_file = make_safe_intermediate_file_path(file, build_dir, change_ext = ".o")
                self.c2o(file, obj_file)
                second_pass_files += [ obj_file ]
            elif cpp_file_p(file):
                has_cxx = True
                obj_file = make_safe_intermediate_file_path(file, build_dir, change_ext = ".o")
                self.cpp2o(file, obj_file)
                second_pass_files += [ obj_file ]
            elif fortran_file_p(file):
                has_fortran = True
                obj_file = make_safe_intermediate_file_path(file, build_dir, change_ext = ".o")
                self.f2o(file, obj_file)
                second_pass_files += [ obj_file ]
            else:
                raise p4a_error("Unsupported extension for input file: " + file)
        
        # Create output files.
        for output_file in output_files:
            output_file = os.path.abspath(os.path.expanduser(output_file))
            
            more_ldflags = []
            
            # Prepare for creating the final binary.
            if lib_file_p(output_file):
                if has_cuda:
                    raise p4a_error("Cannot build a static library when using CUDA")
                more_ldflags += [ "-static" ]
            elif sharedlib_file_p(output_file):
                more_ldflags += [ "-shared" ]
            elif exe_file_p(output_file):
                pass
            else:
                raise p4a_error("I do not know how to make this output file: " + output_file)
            
            final_command = self.cc
            if has_fortran:
                final_command = self.fortran
            elif has_cxx:
                final_command = self.cxx
            
            # Create the final binary.
            run2([ final_command ] + self.ldflags + more_ldflags + [ "-o", output_file ] + second_pass_files + extra_obj,
                extra_env = dict(LD = self.ld, AR = self.ar)
            )
            
            if os.path.exists(output_file):
                done("Generated " + output_file)
            else:
                warn("Expected output file " + output_file + " not found!?")
        

    def parse_include_dirs(self):
        dirs = []
        for opt in self.cppflags:
            if opt.startswith("-I"):
                dirs += [ opt[2:].strip() ]
        return dirs
    
    def parse_lib_dirs(self):
        dirs = []
        for opt in self.cppflags:
            if opt.startswith("-L"):
                dirs += [ opt[2:].strip() ]
        return dirs

    def cmake_write(self, project_name, files, output_file, extra_obj = [], dir = None):
        '''Creates a CMakeLists.txt project file suitable for building the project with CMake.'''
        
        if not project_name:
            project_name = gen_name(prefix = "")
        
        if dir:
            dir = os.path.abspath(dir)
        else:
            dir = os.getcwd()
        if not os.path.isdir(dir):
            os.makedirs(dir)
        cmakelists_file = os.path.join(dir, "CMakeLists.txt")
        
        files += self.extra_files
        
        for file in files:
            if cuda_file_p(file):
                self.cudafy_flags()
        
        global actual_script
        subs = dict(
            script = actual_script,
            time = time.strftime("%Y-%m-%d %H:%M:%S"),
            project = project_name,
            cpp = self.cpp,
            cc = self.cc,
            cxx = self.cxx,
            ld = self.ld,
            ar = self.ar,
            nvcc = self.nvcc,
            fortran = self.fortran,
            cppflags = " ".join(self.cppflags),
            cflags = " ".join(self.cflags),
            ldflags = " ".join(self.ldflags),
            nvccflags = " ".join(self.nvccflags),
            fortranflags = " ".join(self.fortranflags),
            output_dir = os.path.split(output_file[0])[0],
            files = "\n    ".join(files),
            include_dirs = " ".join(self.parse_include_dirs()),
            lib_dirs = " ".join(self.parse_lib_dirs()),
        )
        
        cmakelists = string.Template("""# Generated on $time by $script

cmake_minimum_required(VERSION 2.6)

if(MSVC)
    message(FATAL_ERROR "MSVC not supported yet")
endif()

project("$project")

set(CMAKE_C_COMPILER $cc)
set(CMAKE_CXX_COMPILER $cxx)
set(CMAKE_C_FLAGS $cflags)

set(OUTPUT_DIR $output_dir)
set(LIBRARY_OUTPUT_PATH $${OUTPUT_DIR})
set(CMAKE_BINARY_DIR $${OUTPUT_DIR})
set(EXECUTABLE_OUTPUT_PATH $${OUTPUT_DIR})
set(RUNTIME_OUTPUT_DIRECTORY $${OUTPUT_DIR})
set(ARCHIVE_OUTPUT_DIRECTORY $${OUTPUT_DIR})

set(${project}_HEADER_FILES
#    $${${project}_HEADER_FILES}/header.h
#    Add your headers here.
)

set(${project}_SOURCE_FILES
    $files
)

source_group("Header Files" FILES $${${project}_HEADER_FILES})
source_group("Source Files" FILES $${${project}_SOURCE_FILES})

include_directories($include_dirs)
link_directories($lib_dirs)



""").substitute(subs)

        info(cmakelists)
        
        die("TODO")

        #~ if self.cudafied:
            #~ cmakelist += string.Template("""

#~ """).substitute(subs)

    def cmake_gen(self, build = False):
        pass
    
        

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
