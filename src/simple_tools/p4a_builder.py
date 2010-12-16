#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
# - Ronan Keryell <ronan.keryell@hpc-project.com>
#

'''
Par4All Builder Class
'''

import sys, os, re, shutil, time
from p4a_util import *

actual_script = change_file_ext(os.path.realpath(os.path.abspath(__file__)), ".py", if_ext = ".pyc")
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

#cuda_sdk_dir = ""
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

def get_cuda_cpp_flags():
    return [
        "-I" + os.path.join(get_cuda_dir(), "include")
#        "-I" + os.path.join(get_cuda_sdk_dir(), "C/common/inc"),
    ]

def get_cuda_ld_flags(m64 = True, cutil = False, cublas = False, cufft = False):
    cuda_dir = get_cuda_dir()
#    cuda_sdk_dir = get_cuda_sdk_dir()
    lib_arch_suffix = ""
    if m64:
        lib_arch_suffix = "_x86_64"
    else:
        lib_arch_suffix = "_i386"
    flags = [
        "-L" + os.path.join(cuda_dir, "lib64"),
        "-L" + os.path.join(cuda_dir, "lib"),
#        "-L" + os.path.join(cuda_sdk_dir, "C/lib"),
#        "-L" + os.path.join(cuda_sdk_dir, "C/common/lib/linux"),
        "-Bdynamic", "-lcudart"
    ]
    if cutil:
        flags += [ "-Bstatic", "-lcutil" + lib_arch_suffix ]
    if cublas:
        die("TODO")
    if cufft:
        die("TODO")
    return flags

class p4a_builder:
    """The p4a_builder is used for two main things.
    1 - It keeps track and arrange all the CPP, C, Fortran etc. flags.
    2-  It can buil the program when all the files have been processed by
    PIPS.
    """
    # the lists of flags
    cpp_flags = []
    c_flags = []
    cxx_flags = []
    ld_flags = []
    nvcc_flags = []
    fortran_flags = []

    # the compilers
    cpp = None
    cc = None
    cxx = None
    ld = None
    ar = None
    nvcc = None
    fortran = None

    # extra flags
    m64 = False
    cudafied = False
    extra_source_files = []
    builder = False

    def cudafy_flags(self):
        if self.cudafied:
            return
        self.cpp_flags += get_cuda_cpp_flags()
        self.ld_flags += get_cuda_ld_flags(self.m64)

        self.cpp_flags += [ "-DP4A_ACCEL_CUDA", "-I" + os.environ["P4A_ACCEL_DIR"] ]
        self.extra_source_files += [ os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_accel.cu") ]

        self.cudafied = True

    def __init__(self,
                 cpp_flags = [], c_flags = [], cxx_flags = [], ld_flags = [],
                 nvcc_flags = [], fortran_flags = [],
                 cpp = None, cc = None, cxx = None, ld = None, ar = None,
                 nvcc = None, fortran = None, arch = None,
                 openmp = False, accel_openmp = False, icc = False, cuda = False,mem_optimization=False,
                 add_debug_flags = False, add_optimization_flags = False,
                 no_default_flags = False, build = False
                 ):

        self.builder = build

        if not nvcc:
            nvcc = "nvcc"
        if icc:
            if not which("icc"):
                raise p4a_error("icc is not available -- have you source'd iccvars.sh or iccvars_intel64.sh yet?")
            cc = "icc"
            cxx = "icpc"
            ld = "xild"
            ar = "xiar"
            if which("ifort"):
                fortran = "ifort"
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
                c_flags += [ "-xHOST -O3 -ipo -no-prec-div" ] # Do not specify -fast with implies -static and bugs with STT_GNU_IFUNC upon linkage.
                if fortran == "ifort":
                    fortran_flags += [ "-O3 -ipo" ] # == -fast without -static, same remark as above.
                else:
                    fortran_flags += [ "-O3" ]
            else:
                c_flags += [ "-O3" ]
                fortran_flags += [ "-O3" ]

        if openmp:
            if icc:
                c_flags += [ "-openmp" ]
                fortran_flags += [ "-openmp" ]
                ld_flags += [ "-openmp" ]
            else:
                # Ask for C99 since we generate C99 code:
                c_flags += [ "-std=c99 -fopenmp" ]
                fortran_flags += [ "-fopenmp" ]
                ld_flags += [ "-fopenmp" ]

            if accel_openmp:
                cpp_flags += [ "-DP4A_ACCEL_OPENMP", "-I" + os.environ["P4A_ACCEL_DIR"] ]
                self.extra_source_files += [ os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_accel.c") ]
                
        if mem_optimization:
            self.extra_source_files += [ os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_memory_optimization_runtime.cpp") ]

        if add_debug_flags:
            cpp_flags += [ "-DDEBUG" ] # XXX: does the preprocessor need more definitions?
            c_flags = [ "-g" ] + c_flags
            fortran_flags = [ "-g" ] + fortran_flags

        if not no_default_flags:
            c_flags = [ "-Wall", "-fno-strict-aliasing", "-fPIC" ] + c_flags

        m64 = False
        machine_arch = get_machine_arch()
        if arch is None:
            if machine_arch == "x86_64":
                m64 = True
        else:
            if arch == 32 or arch == "32" or arch == "i386":
                c_flags += [ "-m32" ]
            elif arch == 64 or arch == "64" or arch == "x86_64" or arch == "amd64":
                c_flags += [ "-m64" ]
                m64 = True
            else:
                raise p4a_error("Unsupported architecture: " + arch)

        if c_flags and len(cxx_flags) == 0:
            cxx_flags = c_flags

        self.cpp_flags = cpp_flags
        self.c_flags = c_flags
        self.cxx_flags = cxx_flags
        self.ld_flags = ld_flags
        self.nvcc_flags = nvcc_flags
        self.fortran_flags = fortran_flags

        self.cpp = cpp
        self.cc = cc
        self.cxx = cxx
        self.ld = ld
        self.ar = ar
        self.nvcc = nvcc
        self.fortran = fortran

        self.m64 = m64
        # update cuda flags only if somathing will be built at the end
        if cuda and (self.builder == True):
            self.cudafy_flags()

    def cu2cpp(self, file, output_file):
        run([ self.nvcc, "--cuda" ] + self.cpp_flags + self.nvcc_flags + [ "-o", output_file, file ],
            #extra_env = dict(CPP = self.cpp) # Necessary?
        )

    def c2o(self, file, output_file):
        run([ self.cc, "-c" ] + self.cpp_flags + self.c_flags + [ "-o", output_file, file ])

    def cpp2o(self, file, output_file):
        run([ self.cxx, "-c" ] + self.cpp_flags + self.cxx_flags + [ "-o", output_file, file ])

    def f2o(self, file, output_file):
        run([ self.fortran, "-c" ] + self.cpp_flags + self.fortran_flags + [ "-o", output_file, file ])

    def build(self, files, output_files, extra_obj = [], build_dir = None):

        files += self.extra_source_files

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
            #output_file = os.path.abspath(os.path.expanduser(output_file))

            more_ld_flags = []

            # Prepare for creating the final binary.
            if lib_file_p(output_file):
                if has_cuda:
                    raise p4a_error("Cannot build a static library when using CUDA")
                more_ld_flags += [ "-static" ]
            elif sharedlib_file_p(output_file):
                more_ld_flags += [ "-shared" ]
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
            run([ final_command ] + self.ld_flags + more_ld_flags + [ "-o", output_file ] + second_pass_files + extra_obj,
                extra_env = dict(LD = self.ld, AR = self.ar)
            )

            if os.path.exists(output_file):
                done("Generated " + output_file)
            else:
                warn("Expected output file " + output_file + " not found!?")

    def cmake_write(self, project_name, files, output_files, extra_obj = [], dir = None):
        '''Creates a CMakeLists.txt project file suitable for building the project with CMake.'''

        if not project_name:
            project_name = gen_name(prefix = "")

        # Determine the directory where the CMakeLists.txt file should be put.
        if dir:
            dir = os.path.abspath(dir)
        else:
            dir = os.getcwd()
        if not os.path.isdir(dir):
            os.makedirs(dir)
        cmakelists_file = os.path.join(dir, "CMakeLists.txt")

        # We will try to make all paths relatives to this dir:
        base_dir = dir

        # Make flags CUDA aware if not already the case.
        for file in files:
            if cuda_file_p(file):
                self.cudafy_flags()

        # Append additional required files such as accel files.
        files += self.extra_source_files

        # Make input files relative to the base directory.
        rel_files = []
        for file in files:
            rel_files.append(relativize(file, base_dir))

        # Split input files between regular source files and .cu files,
        # add .cu.cpp files to regular source files for each .cu file.
        cuda_files = []
        cuda_output_files = []
        source_files = []
        header_files = []
        for file in rel_files:
            if cuda_file_p(file):
                cuda_files.append(file)
                cucpp_file = relativize(make_safe_intermediate_file_path(file, base_dir, change_ext = ".cu.cpp"), base_dir)
                cuda_output_files.append(cucpp_file)
                source_files.append(cucpp_file)
            elif header_file_p(file):
                header_files.append(file)
            else:
                source_files.append(file)

        # Create a big dictionary with all the substitutions
        # possible for the string templates.
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
            cpp_flags_all = " ".join(self.cpp_flags),
            cpp_flags = " ".join([elem for elem in self.cpp_flags if not elem.startswith("-I")]),
            c_flags = " ".join(self.c_flags),
            cxx_flags = " ".join(self.cxx_flags),
            ld_flags_all = " ".join(self.ld_flags),
            ld_flags = " ".join([elem for elem in self.ld_flags if not (elem.startswith("-L") or elem.startswith("-l") or elem.startswith("-Bdynamic") or elem.startswith("-Bstatic"))]),
            nvcc_flags = " ".join(self.nvcc_flags),
            fortran_flags = " ".join(self.fortran_flags),
            header_files = "\n    ".join(header_files),
            source_files = "\n    ".join(source_files),
            include_dirs = " ".join([elem[2:].strip() for elem in self.cpp_flags if elem.startswith("-I")]),
            lib_dirs = " ".join([elem[2:].strip() for elem in self.ld_flags if elem.startswith("-L")]),
            libs = " ".join([elem[2:].strip() for elem in self.ld_flags if elem.startswith("-l")]),
        )

        cmakelists = string.Template("""# Generated on $time by $script

cmake_minimum_required(VERSION 2.6)

if(MSVC)
    message(FATAL_ERROR "MSVC not supported yet")
endif()

project("$project")

set(CMAKE_C_COMPILER $cc)
set(CMAKE_CXX_COMPILER $cxx)
set(CMAKE_LINKER $ld)
set(CMAKE_AR $ar)
set(CMAKE_C_FLAGS "$c_flags")
set(CMAKE_CXX_FLAGS "$cxx_flags")
set(CMAKE_LINKER_FLAGS "$ld_flags")

set(${project}_HEADER_FILES
    $header_files
)

set(${project}_SOURCE_FILES
    $source_files
)

source_group("Header Files" FILES $${${project}_HEADER_FILES})
source_group("Source Files" FILES $${${project}_SOURCE_FILES})

include_directories($include_dirs)
link_directories($lib_dirs)
""").substitute(subs)

        for output_file in output_files:
            output_filename = os.path.split(output_file)[1]
            output_dir = relativize(os.path.split(output_file)[0], base_dir)
            subs["output_dir"] = output_dir
            subs["output_filename"] = output_filename
            subs["output_filename_noext"] = os.path.splitext(output_filename)[0]
            if self.cudafied:
                cmakelists += string.Template("\n# CUDA targets for target $output_filename:\n").substitute(subs)
                for i in range(len(cuda_files)):
                    #subs["cuda_target"] = os.path.split(change_file_ext(cuda_files[i], ""))[1]
                    subs["cuda_in"] = cuda_files[i]
                    subs["cuda_out"] = cuda_output_files[i]
                    cmakelists += string.Template("add_custom_command(OUTPUT $cuda_out COMMAND ${nvcc} --cuda ${cpp_flags_all} ${nvcc_flags} -o $cuda_out $cuda_in)\n").substitute(subs)
                    #cmakelists += string.Template("add_dependencies($output_filename $cuda_target)\n").substitute(subs)
            if exe_file_p(output_filename):
                cmakelists += string.Template("""
# Generation of executable target $output_filename:
set(EXECUTABLE_OUTPUT_PATH $output_dir)
add_executable($output_filename_noext $${${project}_SOURCE_FILES})
target_link_libraries($output_filename_noext $libs)
""").substitute(subs)
            elif sharedlib_file_p(output_filename):
                cmakelists += string.Template("""
# Generation of shared library target $output_filename:
set(LIBRARY_OUTPUT_PATH $output_dir)
add_library($output_filename_noext SHARED $${${project}_SOURCE_FILES})
target_link_libraries($output_filename_noext $libs)
set_target_properties($output_filename_noext PROPERTIES DEFINE_SYMBOL "COMPILING_SHARED_LIBRARY")
set_property(TARGET $output_filename_noext PROPERTY LINK_INTERFACE_LIBRARIES "")
""").substitute(subs)
            elif lib_file_p(output_filename):
                cmakelists += string.Template("""
# Generation of static library target $output_filename:
set(ARCHIVE_OUTPUT_DIRECTORY $output_dir)
add_library($output_filename_noext STATIC $${${project}_SOURCE_FILES})
""").substitute(subs)
            else:
                raise p4a_error("I do not know how to build this file type: " + output_file)

        done("Generated " + cmakelists_file)
        write_file(cmakelists_file, cmakelists)

    def cmake_gen(self, dir = None, gen_dir = None, cmake_flags = [], build = False):
         # Determine the directory where the CMakeLists.txt file should be found.
        if dir:
            dir = os.path.abspath(dir)
        else:
            dir = os.getcwd()
        cmakelists_file = os.path.join(dir, "CMakeLists.txt")
        if not os.path.exists(cmakelists_file):
            raise p4a_error("Could not find " + cmakelists_file)
        debug("Generating from " + cmakelists_file)

        # Determine generation directory.
        if not gen_dir:
            gen_dir = os.path.join(os.getcwd(), ".cmake")
        debug("Gen dir: " + gen_dir)
        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)

        run([ "cmake", "." ] + cmake_flags, working_dir = dir)
        if build:
            makeflags = []
            if get_verbosity() >= 2:
                makeflags.append("VERBOSE=1")
            run([ "make" ] + makeflags, working_dir = dir)


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
