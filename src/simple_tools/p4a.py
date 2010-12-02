#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
# - Ronan Keryell <ronan.keryell@hpc-project.com>
#

""" @mainpage Par4All frontend

Par4All frontend implementation
"""

import string, sys, os, re, optparse
from p4a_util import *
from p4a_process import *
from p4a_builder import *
from p4a_git import *
from p4a_version import *
from p4a_opts import *

# To store some arbitrary Python code to be executed inside p4a_process,
# since p4a_process itself is normally executed inside another process:
execute_some_python_code_in_process = None

#some option default values
default_out_suffix = "p4a"
default_out_prefix = ""

def add_own_options(parser):
    "Add the p4a options to the give parser"
    proj_group = optparse.OptionGroup(parser, "Project (aka workspace) options")

    proj_group.add_option("--project-name", "--project", "-p", metavar = "NAME", default = None,
        help = "Name for the project (and for the PIPS workspace database used to work on the analyzed program). If you do not specify the project, a random name will be used.")

    proj_group.add_option("--keep-database", "-k", action = "store_true", default = False,
        help = "Keep database directory after processing.")

    proj_group.add_option("--remove-first", "-r", action = "store_true", default = False,
        help = "Remove existing database directory before processing.")

    parser.add_option_group(proj_group)


    proc_group = optparse.OptionGroup(parser, "PIPS processing options")

    proc_group.add_option("--accel", "-A", action = "store_true", default = False,
        help = "Parallelize for heterogeneous accelerators by using the Par4All Accel run-time that allows executing code for various hardware accelerators such as GPU or even OpenMP emulation.")

    proc_group.add_option("--cuda", "-C", action = "store_true", default = False,
        help = "Enable CUDA generation. Implies --accel.")

    proc_group.add_option("--openmp", "-O", action = "store_true", default = False,
        help = "Parallelize with OpenMP output. If combined with the --accel option, generate Par4All Accel run-time calls and memory transfers with OpenMP implementation instead of native shared-memory OpenMP output. If --cuda is not specified, this option is set by default.")

    proc_group.add_option("--simple", "-S", dest = "simple", action = "store_true", default = False,
        help = "This cancels --openmp and --cuda and does a simple transformation (no parallelization): simply parse the code and regenerate it. Useful to test preprocessor and PIPS intestinal transit")

    proc_group.add_option("--fine", "-F", action = "store_true", default = False,
        help = "Use a fine-grained parallelization algorithm instead of a coarse-grained one.")

    proc_group.add_option("--select-modules", metavar = "REGEXP", default = None,
        help = "Process only the modules (functions and subroutines) whith names matching the regular expression. For example '^saxpy$|dgemm\' will keep only functions or procedures which name is exactly saxpy or contains \"dgemm\". For more information about regular expressions, look at the section 're' of the Python library reference for example. In Fortran, the regex should match uppercase names. Be careful to escape special characters from the shell. Simple quotes are a good way to go for it.")

    proc_group.add_option("--exclude-modules", metavar = "REGEXP", default = None,
        help = "Exclude the modules (functions and subroutines) with names matching the regular expression from the parallelization. For example '(?i)^my_runtime' will skip all the functions or subroutines which names begin with 'my_runtime' in uppercase or lowercase. Have a look to the regular expression documentation for more details.")

    proc_group.add_option("--no-process", "-N", action = "store_true", default = False,
        help = "Bypass all PIPS processing (no parallelizing...) and voids all processing options. The given files are just passed to the back-end compiler. This is merely useful for testing compilation and linking options.")

    proc_group.add_option("--property", "-P", action = "append", metavar = "NAME=VALUE", default = [],
        help = "Define a property for PIPS. Several properties are defined by default (see p4a_process.py). There are many properties in PIPS that can be used to modify its behaviour. Have a look to the 'pipsmake-rc' documentation for their descriptions.")

    proc_group.add_option("--no-spawn", action = "store_true", default = False,
        help = "Do not spawn a child process to run processing (this child process is normally used to post-process the PIPS output and reporting simpler error message for example).")

    parser.add_option_group(proc_group)


    cpp_group = optparse.OptionGroup(parser, "Preprocessing options")

    cpp_group.add_option("--cpp", metavar = "PREPROCESSOR", default = None,
        help = "C preprocessor to use (defaults to gcc -E).")

    cpp_group.add_option("-I", dest="include_dirs", action = "append", metavar = "DIR", default = [],
        help = "Add an include search directory. Same as the compiler -I option. Several are allowed.")

    cpp_group.add_option("-D", dest="defines", action = "append", metavar = "NAME[=VALUE]", default = [],
        help = "Add a preprocessor define. Same as passing the preprocessor a -D option. Several are allowed.")

    cpp_group.add_option("-U", dest="undefines", action = "append", metavar = "NAME", default = [],
        help = "Remove a preprocessor define. Same as passing the preprocessor a -U option. Several are allowed.")

    cpp_group.add_option("--cpp-flags", action = "append", metavar = "FLAGS", default = [],
        help = "Add additional flags for the C preprocessor. Several are allowed.")

    cpp_group.add_option("--skip-recover-includes", action = "store_true", default = False,
        help = "By default, try to recover standard #include. To skip this phase, use this option.")

    cpp_group.add_option("--native-recover-includes", action = "store_true", default = False,
        help = "Use the PyPS default #include recovery method that is less correct if you use complex CPP syntax but is faster. Since it does not rely on the preprocessor that normalized all the included file names, it may be easier to use Par4All in harder context, such as a virtual machine on Windows... By default, use the more complex method of Par4All.")

    parser.add_option_group(cpp_group)


    compile_group = optparse.OptionGroup(parser, "Back-end compilation options")

    compile_group.add_option("--output-file", "--output", "-o", action = "append", metavar = "FILE", default = [],
        help = "This enables automatic compilation of binaries. There can be several of them. Output files can be .o, .so, .a files or have no extension in which case an executable will be built.")

    compile_group.add_option("--exclude-file", "--exclude", "-X", action = "append", metavar = "FILE", default = [],
        help = "Exclude a source file from the back-end compilation. Several are allowed. This is helpful if you need to pass a stub file with dummy function definitions (FFT, linear algebra library...) so that PIPS knows how to parallelize something around them, but do not want this file to end up being compiled since it is not a real implementation. Then use the --extra-file or -l option to give the real implementation to the back-end.")

    compile_group.add_option("--extra-file", "--extra", "-x", action = "append", metavar = "FILE", default = [],
        help = "Include an additional source file when compiling with the back-end compiler. Several are allowed. They will not be processed through PIPS.")

    compile_group.add_option("--cc", metavar = "COMPILER", default = None,
        help = "C compiler to use (defaults to gcc).")

    compile_group.add_option("--cxx", metavar = "COMPILER", default = None,
        help = "C++ compiler to use (defaults to g++).")

    compile_group.add_option("--nvcc", metavar = "COMPILER", default = None,
        help = "NVCC compiler to use (defaults to nvcc). Note that the NVCC compiler is used only to transform .cu files into .cpp files, but not compiling the final binary.")

    compile_group.add_option("--fortran", metavar = "COMPILER", default = None,
        help = "Fortran compiler to use (defaults to gfortran).")

    compile_group.add_option("--ar", metavar = "ARCHIVER", default = None,
        help = "Archiver to use (defaults to ar).")

    compile_group.add_option("--icc", action = "store_true", default = False,
        help = "Automatically switch to Intel's icc/xild/xiar for --cc/--ld/--ar.")

    compile_group.add_option("--debug", "-g", action = "store_true", default = False,
        help = "Add debug flags (-g compiler flag). Have a look to the --no-fast if you want to remove any optimization that would blur the debug.")

    compile_group.add_option("--no-fast", "--not-fast", action = "store_true", default = False,
        help = "Do not add optimized compilation flags automatically.")

    compile_group.add_option("--no-default-flags", action = "store_true", default = False,
        help = "Do not add some C flags such as -fPIC, -g, etc. automatically.")

    compile_group.add_option("--c-flags", action = "append", metavar = "FLAGS", default = [],
        help = "Specify flags to pass to the C compiler. Several are allowed. Note that --cpp-flags will be automatically prepended to the actual flags passed to the compiler.")

    compile_group.add_option("--cxx-flags", action = "append", metavar = "FLAGS", default = [],
        help = "Specify flags to pass to the C++ compiler. Several are allowed. By default, C flags (--c-flags) are also passed to the C++ compiler.")

    compile_group.add_option("--nvcc-flags", action = "append", metavar = "FLAGS", default = [],
        help = "Specify flags to pass to the NVCC compiler. Several are allowed. Note that --cpp-flags will be automatically prepended to the actual flags passed to the compiler.")

    compile_group.add_option("--fortran-flags", action = "append", metavar = "FLAGS", default = [],
        help = "Specify flags to pass to the Fortran compiler. Several are allowed. Note that --cpp-flags will be automatically prepended to the actual flags passed to the compiler.")

    compile_group.add_option("--arch", "-m", metavar = "32|64", default = None,
        help = "Specify compilation target architecture (defaults to current host architecture).")

    compile_group.add_option("--keep-build-dir", "-K", action = "store_true", default = False,
        help = "Do not remove build directory after compilation. If an error occurs, it will not be removed anyways, for further inspection.")

    parser.add_option_group(compile_group)


    link_group = optparse.OptionGroup(parser, "Back-end linking options")

    link_group.add_option("--ld", metavar = "LINKER", default = None,
        help = "Linker to use (defaults to ld).")

    link_group.add_option("-L", dest = "lib_dirs", action = "append", metavar = "DIR", default = [],
        help = "Add a library search directory. Same as the linker -L option. Several are allowed.")

    link_group.add_option("-l", dest = "libs", action = "append",  metavar = "LIB", default = [],
        help = "Specify an input library to link against. Same as the linker -l option. Several are allowed.")

    link_group.add_option("--ld-flags", action = "append", metavar = "FLAGS", default = [],
        help = "Specify additional flags to pass to the linker. Several are allowed.")

    link_group.add_option("--extra-obj", action = "append", metavar = "FILE", default = [],
        help = "Add an additional object file for linking. Several are allowed.")

    parser.add_option_group(link_group)


    cmake_group = optparse.OptionGroup(parser, "CMake file generation options")

    cmake_group.add_option("--cmake", action = "store_true", default = False,
        help = "If output files are specified (with -o), setting this flag will have p4a produce a CMakeLists.txt file in current directory (or in any other directory specified by --cmake-dir). This CMakeLists.txt file will be suitable for building the project with CMake. NB: setting --cmake alone will NOT build the project.")

    cmake_group.add_option("--cmake-flags", action = "append", metavar = "FLAGS", default = [],
        help = "Specify additional flags to pass to CMake. Several are allowed.")

    cmake_group.add_option("--cmake-dir", metavar = "DIR", default = None,
        help = "Output/lookup the CMakeLists.txt file in this directory instead of the current working directory.")

    cmake_group.add_option("--cmake-gen", action = "store_true", default = False,
        help = "If output files are specified (with -o), setting this flag will make p4a try to locate a CMakeLists.txt file in current directory (or in any other directory specified by --cmake-dir), and generate Makefiles in a specific directory (--cmake-gen-dir).")

    cmake_group.add_option("--cmake-gen-dir", metavar = "DIR", default = None,
        help = "Generate Makefiles in this directory instead of <project name>.gen by default.")

    cmake_group.add_option("--cmake-build", action = "store_true", default = False,
        help = "Implies --cmake-gen. Generate Makefiles from the found CMakeLists.txt and run 'make' on them.")

    parser.add_option_group(cmake_group)


    output_group = optparse.OptionGroup(parser, "Output options")

    output_group.add_option("--output-dir", "--od", metavar = "DIR", default = None,
        help = "By default the sources files generated by p4a are located in the folder of the input files. This option allows to generate all the sources output files in the specified directory. When using this option, you can't have files with the same name processed by p4a. An absolute path must be provided, if the directory does not exist p4a will create it. If the output directory is not specified either the output suffix or the output prefix must be set, if not, a suffix will be automatically added to avoid source files destruction.")

    output_group.add_option("--output-suffix", "--os", metavar = "SUF", default = default_out_suffix,
        help = 'Use a suffix to easily recognize files processed by p4a. Default to "' + default_out_suffix + '"')

    output_group.add_option("--output-prefix", "--op", metavar = "PRE", default = default_out_prefix,
        help = 'Use a prefix to easily recognize files processed by p4a. Default to "' + default_out_prefix + '"')

    parser.add_option_group(output_group)

error_re = re.compile(r"^\w+ error ")
warning_re = re.compile(r"^\w+ warning ")
property_redefined_re = re.compile(r"property \S+ redefined")
already_printed_warning_errors = []
began_comment = ""

def pips_output_filter(s):
    '''This callback can be used to filter out lines in PIPS output.
    At minimum verbosity level, we want to display errors/warnings, but not debug messages.'''
    global error_re, warning_re, property_redefined_re, already_printed_warning_errors, began_comment
    if s.find("user warning in c_parse: comment \"") != -1:
        began_comment = s
    elif began_comment:
        began_comment += s
        if s.find("\" is lost") != -1:
            warn("PIPS: " + began_comment)
            already_printed_warning_errors.append(began_comment)
            began_comment = ""
    elif s.find("Cannot preprocess file") != -1:
        error("PIPS: " + s)
    elif error_re.search(s) and s not in already_printed_warning_errors:
        error("PIPS: " + s)
        already_printed_warning_errors.append(s)
    elif warning_re.search(s) and not property_redefined_re.search(s) and s not in already_printed_warning_errors:
        warn("PIPS: " + s)
        already_printed_warning_errors.append(s)
    elif s.find("python3.1: No such file or directory") != -1:
        error("Python 3.1 is required to run the post-processor, please install it")
    elif s.find("p4a_process: ") == 0:
        debug(s, bare = True)
    else:
        debug("PIPS: " + s)


def main():
    '''The function called when this program is executed by its own'''
    parser = optparse.OptionParser(description = __doc__, usage = "%prog [options] [files]; run %prog --help for options")

    # Define all the p4a options:
    add_own_options(parser)
    # Add also all the options common to all the p4a tools:
    add_common_options(parser)

    # Parse the arguments
    (options, args) = parser.parse_args()

    if options.execute:
        local_var = locals()
        # Execute the Python string given by the user:
        exec(options.execute, globals(), local_var)
        # The local variables upwards are not modified
        #print local_var

    if process_common_options(options, args):
        # Delay the PyPS import to be able to give an explicative error message:
        pyps = None
        try:
            pyps = __import__("pyps")
        except:
            pass

        if pyps is None:
            p4a_die_env("Cannot find PyPS!")
        if "P4A_ROOT" not in os.environ:
            p4a_die_env("P4A_ROOT environment variable is not set!")
        if "P4A_ACCEL_DIR" not in os.environ:
            p4a_die_env("P4A_ACCEL_DIR environment variable is not set!")
        if not os.path.isdir(os.environ["P4A_ROOT"]):
            p4a_die_env("Directory pointed by P4A_ROOT environment variable does not exist!")
        if not os.path.isdir(os.environ["P4A_ACCEL_DIR"]):
            p4a_die_env("Directory pointed by P4A_ACCEL_DIR environment variable does not exist!")

        # Check options and set up defaults.
        if len(args) == 0:
            die("Missing input files")

        if options.simple and (options.cuda or options.openmp):
            die("Cannot combine --simple with --cuda and/or --openmp")

        if not options.simple and not options.cuda and not options.openmp:
            info("Defaulting to --openmp")
            options.openmp = True

        if options.cuda and not options.accel:
            info("Enabling --accel because of --cuda")
            options.accel = True

        files = []
        other_files = []
        header_files = []
        # Make all paths absolute for input files, and check passed files extension.
        # Put all files not supported by the p4a_processor class in a separate list.
        for file in args:
            abs_file = os.path.abspath(os.path.expanduser(file))
            if not os.path.exists(abs_file) or not os.path.isfile(abs_file):
                die("Invalid/missing input file: " + abs_file)
            # Check if file has the .p4a suffix, and skip it it is the case:
            if change_file_ext(abs_file, "").endswith(".p4a"):
                warn("Ignoring already processed file: " + file)
                continue
            # Check that a file with the exact same path is not already included:
            if abs_file in files or abs_file in other_files or abs_file in header_files:
                warn("Ignoring second mention of file: " + abs_file)
                continue
            # Check that there is no file with the same name in files
            # to be processed by PIPS (PIPS does not accept several files
            # with same name):
            for review_file in files:
                if os.path.split(review_file)[1] == os.path.split(abs_file)[1]:
                    error(review_file + " has same name as " + abs_file)
                    die("PIPS does not accept several files with same name")
            report_add_file(file)
            ext = get_file_ext(abs_file)
            if c_file_p(file) or fortran_file_p(file):
                files.append(abs_file)
                debug("Input file: " + abs_file)
            elif cxx_file_p(file) or cuda_file_p(file):
                other_files.append(abs_file)
                info("File format not supported by parallelizer, will not be parallelized: " + abs_file)
            elif header_file_p(file):
                header_files.append(abs_file)
                info("Ignoring header file: " + abs_file)
            else:
                die("File format not supported: " + abs_file)

        for file in options.extra_file:
            abs_file = os.path.abspath(os.path.expanduser(file))
            if not os.path.exists(abs_file) or not os.path.isfile(abs_file):
                die("Invalid/missing extra file: " + abs_file)
            # Push the file for a potential report (so that if --report-files is specified, the report
            # will include the input files).
            report_add_file(file)

        # If no project name is provided, try some random names.
        # XXX: would be good to be able to specify the location for the .database and .build dir?
        # Or put it in /tmp by default?..
        project_name = options.project_name
        expected_database_dir = ""
        build_dir = ""
        if not project_name:
            while True:
                project_name = gen_name()
                expected_database_dir = os.path.join(os.getcwd(), project_name + ".database")
                build_dir = os.path.join(os.getcwd(), project_name + ".build")
                if options.remove_first or (not os.path.exists(expected_database_dir) and not os.path.exists(build_dir)):
                    break
            info("Generated project name: " + project_name)
        else:
            expected_database_dir = os.path.join(os.getcwd(), project_name + ".database")
            build_dir = os.path.join(os.getcwd(), project_name + ".build")

        if options.remove_first:
            if os.path.exists(expected_database_dir):
                rmtree(expected_database_dir)
            if os.path.exists(build_dir):
                rmtree(build_dir)

        # Prepare the C preprocessor flags and linker flags.
        cpp_flags = options.cpp_flags
        for include_dir in options.include_dirs:
            cpp_flags += [ "-I" + include_dir ]
        for define in options.defines:
            cpp_flags += [ "-D" + define ]
        for undefine in options.undefines:
            cpp_flags += [ "-U" + undefine ]
        ld_flags = options.ld_flags
        for lib_dir in options.lib_dirs:
            ld_flags += [ "-L" + lib_dir ]
        for lib in options.libs:
            ld_flags += [ "-l" + lib ]

        # Instantiate the builder. It will be used to keep track and arrange all
        # the CPP, C, Fortran etc. flags, apart from being used for building the
        # project after processing, if requested.
        builder = p4a_builder(
            cpp_flags = cpp_flags,
            c_flags = options.c_flags,
            cxx_flags = options.cxx_flags,
            ld_flags = ld_flags,
            nvcc_flags = options.nvcc_flags,
            fortran_flags = options.fortran_flags,
            cpp = options.cpp,
            cc = options.cc,
            cxx = options.cxx,
            ld = options.ld,
            ar = options.ar,
            nvcc = options.nvcc,
            fortran = options.fortran,
            arch = options.arch,
            openmp = options.openmp,
            accel_openmp = options.accel,
            icc = options.icc,
            cuda = options.cuda,
            add_debug_flags = options.debug,
            add_optimization_flags = not options.no_fast,
            no_default_flags = options.no_default_flags
          )

        # TODO: override cpp exe used by pyps/pips with builder.cpp? Not
        # really possible...

        info("CPP flags: " + " ".join(builder.cpp_flags))

        # Process (parallelize) files (or not).
        database_dir = ""
        processed_files = []

        if options.no_process:
            warn("Bypassing PIPS process")
            processed_files = files

        elif len(files) == 0:
            warn("No supported files to process!")

        else:
            # Craft a p4a_processor_input class instance.
            # The class holds all the parameters for the processor (pyps) and for
            # output generation.
            # If --no-spawn is not specified, this instance
            # will be serialized (pickle'd) to ease the
            # passing of parameters to the processor.
            input = p4a_processor_input()
            input.project_name = project_name
            input.accel = options.accel
            input.cuda = options.cuda
            input.openmp = options.openmp
            input.fine = options.fine
            input.select_modules = options.select_modules
            input.exclude_modules = options.exclude_modules
            input.cpp_flags = " ".join(builder.cpp_flags)
            input.files = files
            input.recover_includes = not options.skip_recover_includes
            input.native_recover_includes = options.native_recover_includes
            input.execute_some_python_code_in_process = execute_some_python_code_in_process
            input.output_dir = options.output_dir
            input.output_prefix = options.output_prefix
            input.output_suffix = options.output_suffix

            # Interpret correctly the True/False strings, and integer strings,
            # for the --property option specifications:
            prop_dict = dict()
            for p in options.property:
                (k, v) = p.split("=")
                if v == "False" or v == "false":
                    v = False
                elif v == "True" or v == "true":
                    v = True
                else:
                    try:
                        v = int(v)
                    except:
                        pass
                prop_dict[k] = v

            input.properties = prop_dict

            # This will hold the output (p4a_processor_output instance)
            # when the processor has been called and its output has been
            # deserialized (unpickle'd) (unless --no-spawn is specified in
            # which case the p4a_processor_output instance will be obtained
            # directly):
            output = None

            if options.no_spawn:
                # If --no-spawn is specified, run the processor in the current process:
                # no serialization (pickling) neeed.
                output = process(input)

            else:
                # Else, we are going to serialize the p4a_processor_input instance
                # and deserialize the p4a_processor_output instance when the external
                # processor script has finished.

                # Make temporary files for our input and output "pickles" for the processor script.
                (input_fd, input_file) = tempfile.mkstemp(prefix = "p4a", text = False)
                (output_fd, output_file) = tempfile.mkstemp(prefix = "p4a", text = False)

                # Serialize (pickle) the input parameters for the processor.
                # The processor is a different/separate script, so that we can run it as a different process.
                # We run the processor in a separate process because we want to be able
                # to filter out the PIPS (pyps) output.
                save_pickle(input_file, input)

                # Where is the processor script?
                process_script = os.path.join(get_program_dir(), "p4a_process")

                out, err, ret = "", "", -1
                try:
                    # Do the PIPS job, run the processor script with our input and output pickle files as parameters:
                    out, err, ret = run([ process_script, "--input-file", input_file, "--output-file", output_file ],
                        silent = True,
                        # Do not overload current locale because then we can
                        # no longer work on files with special characters:
                        force_locale = None,
                        stdout_handler = pips_output_filter,
                        stderr_handler = pips_output_filter)
                except:
                    raise p4a_error("PIPS processing aborted")

                # Load the results of the processor from the output pickle file:
                output = load_pickle(output_file)

                # Remove our pickle files:
                os.remove(input_file)
                os.remove(output_file)

            # Assign back useful variables from the processor output:
            processed_files = output.files
            database_dir = output.database_dir

            # If an exception occurred in the processor script (in pyps)
            # it will have been caught and will have been serialized in the
            # processor output class (or put directly in the p4a_processor_output
            # instance if --no-spawn was specified).
            # Raise this exception from our very script if this is the case,
            # so that the normal error catching code is run, so that suggestions
            # are made, so that we can handle --report, etc., etc.
            if output.exception:
                if database_dir:
                    warn("Not removing database directory " + database_dir)
                raise output.exception

        if os.path.isdir(database_dir):
            # Remove database unless otherwise specified.
            if options.keep_database:
                warn("Not removing database directory " + database_dir + " (--keep-database)")
            else:
                # To improve later with a workspace.close() and
                # workspace.delete() some days... -> Yes because some files are left open
                # and we cannot remote the database everytime :-(
                # We should be able to work on an existing database too!
                rmtree(database_dir, can_fail = True)

        for file in processed_files:
            done("Generated " + file, level = 1)
            # Push the file for a potential report (so that if --report-files is specified, the report
            # will include the processed files).
            report_add_file(file)

        if len(options.output_file) == 0:
            if options.cmake or options.cmake_gen or options.cmake_build:
                die("--cmake, --cmake-gen and/or --cmake-build was given but no output files were specified (-o mybinary, -o mysharedlib.so etc.)")
            # Build not requested.
            return

        all_buildable_files = []

        # Filter out the excluded files from the build (--exclude-file):
        for file in processed_files + other_files + options.extra_file:
            # This is normally not necessary at this point, but just to be sure:
            file = os.path.abspath(os.path.expanduser(file))
            found = False
            for exclude_file in options.exclude_file:
                abs_exclude_file = os.path.abspath(os.path.expanduser(exclude_file))
                if abs_exclude_file == file:
                    warn("Excluding " + file + " from the build (--exclude-file " + exclude_file + ")")
                    found = True
                else:
                    debug("Compared " + file + " to --exclude-file " + exclude_file + " -> " + abs_exclude_file + ", no match")
            if not found:
                all_buildable_files.append(file)

        if len(all_buildable_files) == 0:
            die("No buildable input files!")

        # Make every path absolute.
        output_files = []
        for file in options.output_file:
            output_files.append(os.path.abspath(os.path.expanduser(file)))

        # Generate CMakeLists.txt/build using it as requested.
        if options.cmake or options.cmake_gen or options.cmake_build:
            # Push the CMakeLists.txt file for a potential report
            # (so that if --report-files is specified, the report
            # will include this file).
            report_add_file(os.path.join(options.cmake_dir, "CMakeLists.txt"))
            if options.cmake:
                builder.cmake_write(project_name, all_buildable_files + header_files,
                    output_files, extra_obj = options.extra_obj, dir = options.cmake_dir)
            if options.cmake_gen or options.cmake_build:
                builder.cmake_gen(dir = options.cmake_dir, gen_dir = options.cmake_gen_dir,
                    cmake_flags = options.cmake_flags, build = options.cmake_build)
            return

        # All the building is handled by p4a_builder.py:
        try:
            info("Building " + ", ".join(output_files))
            builder.build(files = all_buildable_files, output_files = output_files,
                extra_obj = options.extra_obj, build_dir = build_dir)
        except:
            warn("Build directory was not removed: " + build_dir)
            raise

        # Remove build dir as requested.
        if os.path.isdir(build_dir):
            if options.keep_build_dir:
                warn("Not removing build directory " + build_dir + " (--keep-build-dir)")
            else:
                rmtree(build_dir, can_fail = True)

# If this file is called as a script, execute the main:
if __name__ == "__main__":
    exec_and_deal_with_errors(main)

# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
