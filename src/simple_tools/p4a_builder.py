#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
# - Ronan Keryell <ronan.keryell@hpc-project.com>
#
import p4a_util
import string
import os
import time

'''
Par4All Builder Class
'''


actual_script = p4a_util.change_file_ext(os.path.realpath(os.path.abspath(__file__)), ".py", if_ext = ".pyc")
script_dir = os.path.split(actual_script)[0]

def make_safe_intermediate_file_path(input_file, build_dir, change_ext = None):
    '''Make up a safe intermediate file path.'''
    intermediate_file = ""
    while True:
        intermediate_file = p4a_util.file_add_suffix(os.path.join(build_dir, os.path.split(input_file)[1]), "_" + p4a_util.gen_name(prefix = ""))
        if change_ext:
            intermediate_file = p4a_util.change_file_ext(intermediate_file, change_ext)
        if not os.path.exists(intermediate_file):
            break
    return intermediate_file

# By default, no specific value for CUDA_DIR defining where CUDA is
# installed.  Note that if there is no need for CUDA or if the CUDA
# environment is installed with standard packages on the OS (for example
# package nvidia-cuda-toolkit on Debian), this can remain unset:
cuda_dir = None
def get_cuda_dir():
    global cuda_dir
    if not cuda_dir:
        if "CUDA_DIR" in os.environ:
            cuda_dir = os.path.expanduser(os.environ["CUDA_DIR"])
        else:
            cuda_dir = "/usr/local/cuda"
            p4a_util.warn("CUDA_DIR environment variable undefined. Using '" +
                cuda_dir + "' as default location for CUDA installation")
        if not os.path.isdir(cuda_dir):
            p4a_util.warn("CUDA directory not found or invalid (" + cuda_dir
                + "). This may work if it is already installed in a standard place already searched by the system tools. If not, please set the CUDA_DIR environment variable correctly")
    return cuda_dir


amd_opencl_dir = None
def get_amd_opencl_dir():
    global amd_opencl_dir
    if not amd_opencl_dir:
        if "AMDAPPSDKROOT" in os.environ:
            amd_opencl_dir = os.path.expanduser(os.environ["AMDAPPSDKROOT"])
    return amd_opencl_dir


#cuda_sdk_dir = ""
def get_cuda_sdk_dir():
    global cuda_sdk_dir
    if not cuda_sdk_dir:
        if "CUDA_SDK_DIR" in os.environ:
            cuda_sdk_dir = os.path.expanduser(os.environ["CUDA_SDK_DIR"])
        else:
            cuda_sdk_dir = os.path.expanduser("~/NVIDIA_GPU_Computing_SDK")
            p4a_util.warn("CUDA_SDK_DIR environment variable undefined. Using '" +
                cuda_sdk_dir + "' as default location for CUDA installation")
        if not os.path.isdir(cuda_sdk_dir):
            raise p4a_util.p4a_error("CUDA SDK directory not found or invalid (" + cuda_sdk_dir
                + "). Please set the CUDA_SDK_DIR environment variable correctly")
    return cuda_sdk_dir

def get_opencl_cpp_flags():
    '''To be restructured for multi-platforms OpenCL
	'''
    opencl_dir=get_amd_opencl_dir()
    if opencl_dir:
        opencl_flags=["-I"+os.path.join(opencl_dir, "include")]
    else:
        opencl_dir=get_cuda_dir()
        if opencl_dir:
            opencl_flags=["-I" + os.path.join(opencl_dir, "include/CL")
                          + " -I" + os.path.join(opencl_dir, "include/CL2")
                          + " -I" +  os.path.join(opencl_dir, "include")
                          ]
        else:
            opencl_flags = []
    return opencl_flags


def get_cuda_cpp_flags():
    if get_cuda_dir():
        return [
            "-I" + os.path.join(get_cuda_dir(), "include")
            # "-I" + os.path.join(get_cuda_sdk_dir(), "C/common/inc"),
            ]
    else:
        return []


def get_cuda_ld_flags(m64 = True, cutil = False, cublas = False, cufft = False):
#    cuda_sdk_dir = get_cuda_sdk_dir()
    lib_arch_suffix = ""
    if m64:
        lib_arch_suffix = "_x86_64"
    else:
        lib_arch_suffix = "_i386"
    flags = [ "-Bdynamic", "-lcudart" ]
    # Insert specific directory paths if specified:
    cuda_dir = get_cuda_dir()
    if cuda_dir:
        flags = [
            "-L" + os.path.join(cuda_dir, "lib64"),
            "-L" + os.path.join(cuda_dir, "lib"),
            # "-L" + os.path.join(cuda_sdk_dir, "C/lib"),
            # "-L" + os.path.join(cuda_sdk_dir, "C/common/lib/linux"),
            ] + flags
    if cutil:
        flags += [ "-Bstatic", "-lcutil" + lib_arch_suffix ]
    if cublas:
        p4a_util.die("TODO")
    if cufft:
        p4a_util.die("TODO")
    return flags


def get_opencl_ld_flags(m64 = True):
    '''To be restructured for multi-platforms OpenCL
    '''
    lib_opencl_path = None
    opencl_dir=get_amd_opencl_dir()
    lib_arch_suffix = ""
    if opencl_dir:
        lib_opencl_path=os.path.join(opencl_dir, "lib/")
        if m64:
            lib_arch_suffix = "x86_64"
        else:
            lib_arch_suffix = "x86"
    else:
        opencl_dir = get_cuda_dir()
        if opencl_dir:
            lib_opencl_path=os.path.join(opencl_dir, "lib")
            if m64:
                lib_arch_suffix = "64"
            else:
                lib_arch_suffix = ""
    if lib_opencl_path:
        flags = [
            # "-L" + os.path.join(cuda_dir, "lib64"),
            "-L" + os.path.join(opencl_dir, lib_opencl_path + lib_arch_suffix),
            # "-L" + os.path.join(cuda_dir, "lib"),  # for cuda
            "-L" + lib_opencl_path,
            "-L/usr/lib",
            "-l OpenCL"
            ]
    else:
        flags = []
    return flags


class p4a_builder:
    """The p4a_builder is used for two main things.
    1 - It keeps track and arrange all the CPP, C, Fortran etc. flags.
    2-  It can build the program when all the files have been processed by
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
    openclified = False
    extra_source_files = []
    builder = False

    def cudafy_flags(self):
        if self.cudafied:
            return
        self.cpp_flags += get_cuda_cpp_flags()
        self.ld_flags += get_cuda_ld_flags(self.m64)

        self.cpp_flags += [ "-DP4A_ACCEL_CUDA", "-I" + os.environ["P4A_ACCEL_DIR"] ]
        self.extra_source_files += [ os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_accel.cu") ]
        self.fortran_flags.append ("-ffree-line-length-none")
        self.cudafied = True

    def openclify_flags(self):
        if self.openclified:
            return
        self.cpp_flags += get_opencl_cpp_flags()
        self.ld_flags += get_opencl_ld_flags(self.m64)

        self.cpp_flags += [ "-DP4A_ACCEL_OPENCL", "-I" + os.environ["P4A_ACCEL_DIR"] ]
        self.extra_source_files += [ os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_accel.c") ]
        self.fortran_flags.append ("-ffree-line-length-none")
        self.openclified = True

    def __init__(self,
                 cpp_flags = [], c_flags = [], cxx_flags = [], ld_flags = [],
                 ld_libs = [],
                 nvcc_flags = [], fortran_flags = [],
                 cpp = None, cc = None, cxx = None, ld = None, ar = None,
                 nvcc = None, fortran = None, arch = None,
                 openmp = False, accel_openmp = False, icc = False,
                 cuda = False, opencl = False, atomic=False,kernel_unroll=0,
                 com_optimization=False,cuda_cc=2,fftw3=False,
                 add_debug_flags = False, add_optimization_flags = False,
                 add_openmp_flag = False, no_default_flags = False,
                 build = False
                 ):

        self.builder = build

        if atomic:
            if cuda_cc == 1:
                raise p4a_util.p4a_error("Atomic operations isn't available with CUDA compute capability 1.0?")


        if cuda_cc == 1:
            nvcc_flags = [ "-arch=sm_10" ] + nvcc_flags
        elif cuda_cc == 1.1:
            nvcc_flags = [ "-arch=sm_11" ] + nvcc_flags
        elif cuda_cc == 1.2:
            nvcc_flags = [ "-arch=sm_12" ] + nvcc_flags
        elif cuda_cc == 1.3:
            nvcc_flags = [ "-arch=sm_13" ] + nvcc_flags
        elif cuda_cc == 2.0:
            nvcc_flags = [ "-arch=sm_20" ] + nvcc_flags

        if not nvcc:
            nvcc = "nvcc"
        if icc:
            if not p4a_util.which("icc"):
                raise p4a_util.p4a_error("icc is not available -- have you source'd iccvars.sh or iccvars_intel64.sh yet?")
            cc = "icc"
            cxx = "icpc"
            ld = "xild"
            ar = "xiar"
            if p4a_util.which("ifort"):
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

        # We need this to make sure nvcc uses the compiler given by the user
        nvcc_flags.append("-ccbin="+cxx)

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
            # Ask for C99 since PIPS generates C99 code:
            c_flags.append ("-std=gnu99")
            if add_openmp_flag:
                if icc:
                    c_flags += [ "-openmp" ]
                    fortran_flags += [ "-openmp" ]
                    ld_flags += [ "-openmp" ]
                else:
                    c_flags += [ "-fopenmp" ]
                    fortran_flags += [ "-fopenmp" ]
                    ld_flags += [ "-fopenmp" ]
            else:
                if icc:
                    c_flags += [ "-openmp-stubs" ]
                    fortran_flags += [ "-openmp-stubs" ]
                    ld_flags += [ "-openmp-stubs" ]
                else:
                    c_flags += [ "-fno-openmp" ]
                    fortran_flags += [ "-fno-openmp" ]
                    ld_flags += [ "-fno-openmp" ]

            if accel_openmp:
                cpp_flags += [ "-DP4A_ACCEL_OPENMP", "-I" + os.environ["P4A_ACCEL_DIR"] ]
                fortran_flags.append ("-ffree-line-length-none")
                self.extra_source_files += [ os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_accel.c") ]

        if com_optimization:
            self.extra_source_files += [ os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_communication_optimization_runtime.cpp") ]
            cpp_flags += [ "-DP4A_COMMUNICATION_RUNTIME" ]

        if fftw3:
            cpp_flags += [ "-DP4A_RUNTIME_FFTW", "-I" + os.environ["P4A_ACCEL_DIR"] ]
            self.extra_source_files += [ os.path.join(os.environ["P4A_ACCEL_DIR"], "p4a_fftw3_runtime.cpp") ]
            if cuda :
                ld_libs += [ "-lcufft" ]
            elif opencl :
                p4a_util.die("fftw3+opencl :  TODO")
            else :
                ld_libs += [ "-lfftw3 -lfftw3f" ]



        if add_debug_flags:
            cpp_flags += [ "-DDEBUG" ] # XXX: does the preprocessor need more definitions?
            c_flags = [ "-g" ] + c_flags
            fortran_flags = [ "-g" ] + fortran_flags
            nvcc_flags = [ "-g" ] + [ "-G3" ] + nvcc_flags

        if not no_default_flags:
            c_flags = [ "-Wall", "-fno-strict-aliasing", "-fPIC" ] + c_flags

        m64 = False
        machine_arch = p4a_util.get_machine_arch()
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
                raise p4a_util.p4a_error("Unsupported architecture: " + arch)

        if c_flags and len(cxx_flags) == 0:
            cxx_flags = c_flags

        self.cpp_flags = cpp_flags
        self.c_flags = c_flags
        self.cxx_flags = cxx_flags
        self.ld_flags = ld_flags
        self.ld_libs = ld_libs
        self.nvcc_flags = nvcc_flags
        self.fortran_flags = fortran_flags

        self.cpp = cpp
        self.cc = cc
        self.cxx = cxx
        self.ld = ld
        self.ar = ar
        self.nvcc = nvcc
        self.fortran = fortran
        self.opencl = opencl

        self.m64 = m64
        # update cuda flags only if somathing will be built at the end
        if cuda and (self.builder == True):
            self.cudafy_flags()

        # update opencl flags only if something will be built at the end
        if opencl and (self.builder == True):
            self.openclify_flags()
            c_flags.append ("-std=gnu99")

    def cu2cpp(self, file, output_file):
        p4a_util.run([ self.nvcc, "--cuda" ] + self.cpp_flags + self.nvcc_flags + [ "-o", output_file, file],
            #extra_env = dict(CPP = self.cpp) # Necessary?
        )

    def c2o(self, file, output_file):
        p4a_util.run([ self.cc, "-c" ] + self.cpp_flags + self.c_flags + [ "-o", output_file, file ])

    def cpp2o(self, file, output_file):
        p4a_util.run([ self.cxx, "-c" ] + self.cpp_flags + self.cxx_flags + [ "-o", output_file, file ])

    def f2o(self, file, output_file):
        p4a_util.run([ self.fortran, "-c" ] + self.cpp_flags + self.fortran_flags + [ "-o", output_file, file ])

    def build(self, files, output_files, extra_obj = [], build_dir = None):
        """ Build progams, libraries, objects or ...
        files, the files to be used by the building process
        output_files, the files to be produced
        extra_obj, some exta objects to be used by the building process
        build_dir, the build directory
        """
        files += self.extra_source_files

        has_cuda = False
        has_opencl = False
        has_c = False
        has_cxx = False
        has_fortran = False

        # Determine build directory.
        if not build_dir:
            build_dir = os.path.join(os.getcwd(), ".build")
        p4a_util.debug("Build dir: " + build_dir)
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)

        # the build_dir has to be added to the compiler search path in fortran
        # because .mod file are produce there
        self.fortran_flags.append ("-J " + build_dir)

        # First pass: make .c, .cpp or .f files out of other extensions (.cu, ..):
        first_pass_files = []
        for file in files:
            if p4a_util.cuda_file_p(file):
                has_cuda = True
                self.cudafy_flags()
                cucpp_file = make_safe_intermediate_file_path(file, build_dir, change_ext = ".cu.cpp")
                self.cu2cpp(file, cucpp_file)
                first_pass_files += [ cucpp_file ]
            elif p4a_util.opencl_file_p(file):
                has_opencl = True
                self.openclify_flags()
            else:
                first_pass_files += [ file ]

        # Second pass: make object files out of source files.
        second_pass_files = []
        for file in first_pass_files:
            if p4a_util.c_file_p(file):
                has_c = True
                obj_file = p4a_util.quote_fname(make_safe_intermediate_file_path(file, build_dir, change_ext = ".o"))
                self.c2o(p4a_util.quote_fname(file),obj_file)
                second_pass_files += [ obj_file ]
            elif p4a_util.cpp_file_p(file):
                has_cxx = True
                obj_file = p4a_util.quote_fname(make_safe_intermediate_file_path(file, build_dir, change_ext = ".o"))
                self.cpp2o(p4a_util.quote_fname(file), obj_file)
                second_pass_files += [ obj_file ]
            elif p4a_util.fortran_file_p(file):
                has_fortran = True
                obj_file = p4a_util.quote_fname(make_safe_intermediate_file_path(file, build_dir, change_ext = ".o"))
                self.f2o(p4a_util.quote_fname(file), obj_file)
                second_pass_files += [ obj_file ]
            else:
                raise p4a_util.p4a_error("Unsupported extension for input file: " + file)

        # Create output files.
        for output_file in output_files:
            #output_file = os.path.abspath(os.path.expanduser(output_file))

            more_ld_flags = []

            # Prepare for creating the final binary.
            if p4a_util.lib_file_p(output_file):
                if has_cuda:
                    raise p4a_util.p4a_error("Cannot build a static library when using CUDA")
                if has_opencl:
                    raise p4a_util.p4a_error("Cannot build a static library when using OPENCL")
                more_ld_flags += [ "-static" ]
            elif p4a_util.sharedlib_file_p(output_file):
                more_ld_flags += [ "-shared" ]
            elif p4a_util.exe_file_p(output_file):
                pass
            else:
                raise p4a_util.p4a_error("I do not know how to make this output file: " + output_file)

            final_command = self.cc
            if has_fortran:
                final_command = self.fortran
            elif has_cxx:
                final_command = self.cxx

            # Quote extra_obj
            quoted_extra_obj=[]
            for obj in extra_obj:
				quoted_extra_obj+=[p4a_util.quote_fname(obj)]
            # Create the final binary.
            p4a_util.run([ final_command ] + [ "-o", p4a_util.quote_fname(output_file) ] + second_pass_files + quoted_extra_obj + self.ld_flags + more_ld_flags + self.ld_libs,
                extra_env = dict(LD = self.ld, AR = self.ar)
            )

            if os.path.exists(output_file):
                p4a_util.done("Generated " + output_file)
            else:
                p4a_util.warn("Expected output file " + output_file + " not found!?")

    def cmake_write(self, project_name, files, output_files, extra_obj = [], dir = None):
        '''Creates a CMakeLists.txt project file suitable for building the project with CMake.'''

        if not project_name:
            project_name = p4a_util.gen_name(prefix = "")

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
            if p4a_util.cuda_file_p(file):
                self.cudafy_flags()
            if p4a_util.opencl_file_p(file):
                self.openclify_flags()

        # Append additional required files such as accel files.
        files += self.extra_source_files

        # Make input files relative to the base directory.
        rel_files = []
        for file in files:
            rel_files.append(os.path.abspath(file))

        # Split input files between regular source files and .cu files,
        # add .cu.cpp files to regular source files for each .cu file.
        cuda_files = []
        cuda_output_files = []
        opencl_files = []
        source_files = []
        header_files = []
        for file in rel_files:
            if p4a_util.cuda_file_p(file):
                cuda_files.append(file)
                cucpp_file = os.path.abspath(make_safe_intermediate_file_path(file, base_dir, change_ext = ".cu.cpp"))
                cuda_output_files.append(cucpp_file)
                source_files.append(cucpp_file)
            elif p4a_util.opencl_file_p(file):
                opencl_files.append(file)
            elif p4a_util.header_file_p(file):
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
            libs = " ".join([elem[2:].strip() for elem in self.ld_libs if elem.startswith("-l")]),
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
set(CMAKE_C_FLAGS "$cpp_flags $c_flags")
set(CMAKE_CXX_FLAGS "$cpp_flags $cxx_flags")
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
            output_dir = p4a_util.relativize(os.path.split(output_file)[0], base_dir)
            subs["output_dir"] = output_dir
            subs["output_filename"] = output_filename
            subs["output_filename_noext"] = os.path.splitext(output_filename)[0]
            if self.cudafied:
                cmakelists += string.Template("\n# CUDA targets for target $output_filename:\n").substitute(subs)
                for i in range(len(cuda_files)):
                    #subs["cuda_target"] = os.path.split(p4a_util.change_file_ext(cuda_files[i], ""))[1]
                    subs["cuda_in"] = cuda_files[i]
                    subs["cuda_out"] = cuda_output_files[i]
                    cmakelists += string.Template("add_custom_command(OUTPUT $cuda_out COMMAND ${nvcc} --cuda ${cpp_flags_all} ${nvcc_flags} -o $cuda_out $cuda_in)\n").substitute(subs)
                    #cmakelists += string.Template("add_dependencies($output_filename $cuda_target)\n").substitute(subs)
            if p4a_util.exe_file_p(output_filename):
                cmakelists += string.Template("""
# Generation of executable target $output_filename:
set(EXECUTABLE_OUTPUT_PATH $output_dir)
add_executable($output_filename_noext $${${project}_SOURCE_FILES})
target_link_libraries($output_filename_noext $libs)
""").substitute(subs)
            elif p4a_util.sharedlib_file_p(output_filename):
                cmakelists += string.Template("""
# Generation of shared library target $output_filename:
set(LIBRARY_OUTPUT_PATH $output_dir)
add_library($output_filename_noext SHARED $${${project}_SOURCE_FILES})
target_link_libraries($output_filename_noext $libs)
set_target_properties($output_filename_noext PROPERTIES DEFINE_SYMBOL "COMPILING_SHARED_LIBRARY")
set_property(TARGET $output_filename_noext PROPERTY LINK_INTERFACE_LIBRARIES "")
""").substitute(subs)
            elif p4a_util.lib_file_p(output_filename):
                cmakelists += string.Template("""
# Generation of static library target $output_filename:
set(ARCHIVE_OUTPUT_DIRECTORY $output_dir)
add_library($output_filename_noext STATIC $${${project}_SOURCE_FILES})
""").substitute(subs)
            else:
                raise p4a_util.p4a_error("I do not know how to build this file type: " + output_file)

        p4a_util.done("Generated " + cmakelists_file)
        p4a_util.write_file(cmakelists_file, cmakelists)

    def cmake_gen(self, dir = None, cmake_flags = [], build = False):
         # Determine the directory where the CMakeLists.txt file should be found.
        if dir:
            dir = os.path.abspath(dir)
        else:
            dir = os.getcwd()
        cmakelists_file = os.path.join(dir, "CMakeLists.txt")
        if not os.path.exists(cmakelists_file):
            raise p4a_util.p4a_error("Could not find " + cmakelists_file)
        p4a_util.debug("Generating from " + cmakelists_file)

        p4a_util.run([ "cmake", "." ] + cmake_flags, working_dir = dir)
        if build:
            makeflags = []
            if p4a_util.get_verbosity() >= 2:
                makeflags.append("VERBOSE=1")
            p4a_util.run([ "make" ] + makeflags, working_dir = dir)

    def append_cpp_flags (self, flag):
        self.cpp_flags.append (flag)

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
