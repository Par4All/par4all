#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
# - Ronan Keryell <ronan.keryell@hpc-project.com>
#

'''
Par4All Frontend Script
'''

import string, sys, os, re, optparse
from p4a_util import *
from p4a_processor import *
from p4a_builder import *
from p4a_git import *
from p4a_version import *

def main(options = {}, args = []):

    pyps = None
    try:
        pyps = __import__("pyps")
    except:
        pass

    if (pyps is None
        or "P4A_ROOT" not in os.environ 
        or "P4A_ACCEL_DIR" not in os.environ 
        or not os.path.exists(os.environ["P4A_ROOT"]) 
        or not os.path.exists(os.environ["P4A_ACCEL_DIR"])):
        die("The Par4All environment has not been properly set (through par4all-rc.sh)")

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
        if not os.path.exists(abs_file):
            die("Invalid/missing input file: " + abs_file)
        # Check if file has the .p4a suffix, and skip it it is the case:
        if change_file_ext(abs_file, "").endswith(".p4a"):
            warn("Ignoring already processed file: " + file)
            continue
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
    
    ###################
    ### XXXXXXXXXXXXXX TODO: override CPP used by the processor -> pyps -> pips with builder.cpp
    
    info("CPP flags: " + " ".join(builder.cpp_flags))

    # Process (parallelize) files (or not).
    database_dir = ""
    processor_output_files = []
    if options.no_process:
        warn("Bypassing processor")
        processor_output_files = files
    elif len(files) == 0:
        warn("No supported files to process!")
    else:
        try:
            # Create a workspace with PIPS:
            processor = p4a_processor(files = files,
                                  project_name = project_name,
                                  verbose = (verbosity != 0),
                                  cpp_flags = " ".join(builder.cpp_flags),
                                  recover_includes = not options.skip_recover_includes,
                                  filter_include = options.include_modules,
                                  filter_exclude = options.exclude_modules,
                                  accel = options.accel,
                                  cuda = options.cuda)
            
            # Save it for later.
            database_dir = os.path.abspath(processor.workspace.directory())

            # First apply some generic parallelization:
            processor.parallelize(options.fine)

            if options.cuda:
                processor.gpuify()

            if options.openmp:
                processor.ompify()
            
            # Write the output files.
            processor_output_files = processor.save()
            
            del processor
            
        except p4a_error:
            error("Processing of " + ", ".join(files) + " failed: " + sys.exc_info()[1].msg)
            if database_dir:
                error("Database directory was " + database_dir)
            die("Aborting")

    if os.path.isdir(database_dir):
        # Remove database unless otherwise specified.
        if options.keep_database:
            warn("Not removing database directory " + database_dir) 
        else:
            # To improve later with a workspace.close() and
            # workspace.delete() some days... -> Yes because some files are left open
            # and we cannot remote the database everytime :-(
            # We should be able to work on an existing database too!
            rmtree(database_dir, can_fail = True)

    if len(options.output_file) == 0:
        # Build not requested.
        return

    all_buildable_files = processor_output_files + other_files + options.extra
    if len(all_buildable_files) == 0:
        die("No buildable input files")
    
    # Make every path absolute.
    output_files = []
    for file in options.output_file:
        output_files.append(os.path.abspath(os.path.expanduser(file)))

    # Generate CMakeLists.txt/build using it as requested.
    if options.cmake or options.cmake_gen or options.cmake_build:
        if options.cmake:
            builder.cmake_write(project_name, all_buildable_files + header_files, 
                output_files, extra_obj = options.extra_obj, dir = options.cmake_dir)
        if options.cmake_gen or options.cmake_build:
            builder.cmake_gen(dir = options.cmake_dir, gen_dir = options.cmake_gen_dir, 
                cmake_flags = options.cmake_flags, build = options.cmake_build)
        return
    
    try:
        info("Building " + ", ".join(output_files))
        builder.build(files = all_buildable_files, output_files = output_files, 
            extra_obj = options.extra_obj, build_dir = build_dir)
    except p4a_error:
        error("Building failed: " + sys.exc_info()[1].msg)
        error("Build directory was " + build_dir)
        die("Aborting")

    # Remove build dir as requested.
    if os.path.isdir(build_dir):
        if options.keep_build_dir:
            warn("Not removing build directory " + build_dir)
        else:
            rmtree(build_dir, can_fail = True)


# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
