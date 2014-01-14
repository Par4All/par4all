#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
# - Ronan Keryell <ronan.keryell@hpc-project.com>
#
import p4a_util
import optparse
import subprocess
import sys
import os
import re
import shutil
import p4a_processor
import p4a_scmp_compiler
import p4a_spear_processor

'''
Par4All processing
'''

def add_module_options(parser):
    "Add to an optparse option parser the options related to this module"
    group = optparse.OptionGroup(parser, "Processor Options")

    group.add_option("--input-file", metavar = "FILE", default = None,
        help = "Input file (as created using the pickle module on a p4a_processor_input instance).")

    group.add_option("--output-file", metavar = "FILE", default = None,
        help = "Output file (to be created using the pickle module on a p4a_processor_output instance).")

    parser.add_option_group(group)


def process(input):
    """Process the input files with PIPS and return the list of produced files

    The aim of this method is mainly to invoke PIPS in another process
    instead of the current one, according to
    input.execute_some_python_code_in_process
    """
    output = p4a_processor.p4a_processor_output()

    # Execute some arbitrary Python code here if asked:
    if input.execute_some_python_code_in_process:
        exec(input.execute_some_python_code_in_process)

    try:

        if input.scmp:
            # scmp case
            processor = p4a_scmp_compiler.p4a_scmp_compiler(
                project_name = input.project_name,
                verbose = True,
                files = input.files
            )
            output.database_dir = processor.get_database_directory()
            processor.go()
            output.files = processor.get_generated_files()
        else:
            # Create a workspace with PIPS:
            if input.spear:
                processor_class=p4a_spear_processor.p4a_spear_processor
            else:
                processor_class=p4a_processor.p4a_processor

            processor = processor_class(
                project_name = input.project_name,
                cpp_flags = input.cpp_flags,
                verbose = True,
                files = input.files,
                filter_select = input.select_modules,
                filter_exclude = input.exclude_modules,
                noalias = input.noalias,
                pointer_analysis = input.pointer_analysis,
                accel = input.accel,
                cuda = input.cuda,
                opencl = input.opencl,
                openmp=input.openmp,
                spear=input.spear,
                com_optimization = input.com_optimization,
                cuda_cc = input.cuda_cc,
                fftw3 = input.fftw3,
                c99 = input.c99,
                atomic = input.atomic,
                kernel_unroll = input.kernel_unroll,
                use_pocc = input.pocc,
                pocc_options = input.pocc_options,
                recover_includes = input.recover_includes,
                native_recover_includes = input.native_recover_includes,
                properties = input.properties,
                apply_phases = input.apply_phases,
                brokers = input.brokers
            )
            if input.accel:
                p4a_util.warn("Activating fine-grain parallelization for accelerator mode")
                input.fine_grain = True

            output.database_dir = processor.get_database_directory()

            # First apply some generic parallelization:
            processor.parallelize(fine_grain = input.fine_grain,
                                  apply_phases_before = input.apply_phases['abp'],
                                  apply_phases_after = input.apply_phases['aap'],
                                  omp=input.openmp and not input.accel)

            if input.accel:
                # Generate code for a GPU-like accelerator. Note that we can
                # have an OpenMP implementation of it if OpenMP option is set
                # too:
                processor.gpuify(
                        fine_grain = input.fine_grain,
                        apply_phases_kernel = input.apply_phases['akg'],
                        apply_phases_kernel_launcher = input.apply_phases['aklg'],
                        apply_phases_wrapper = input.apply_phases['awg'],
                        apply_phases_after = input.apply_phases['aag'])

            if input.openmp and not input.accel:
                # Parallelize the code in an OpenMP way:
                processor.ompify(
                        apply_phases_before = input.apply_phases['abo'],
                        apply_phases_after = input.apply_phases['aao'])

            # Write the output files.
            output.files = processor.save(input.output_dir,
                                          input.output_prefix,
                                          input.output_suffix)

    except:
        # Get the exception description:
        e = sys.exc_info()
        exception = e[1]
        if exception.__class__.__name__ == "RuntimeError" and str(exception).find("pips") != -1:
            output.exception = p4a_util.p4a_error("An error occurred in PIPS while processing " + ", ".join(input.files))
        else:
            # Since the traceback object cannot be serialized, display
            # here the exec_info for more information on stderr:
            sys.excepthook(*e)
            # And only push the exception further for information:
            output.exception = exception

    return output


def process_file(input_file, output_file):
    input = p4a_util.load_pickle(input_file)
    output = process(input)
    p4a_util.save_pickle(output_file, output)


def main(options, args = []):
    """Process the workspace with PIPS

    The description of the input and the output are given as 2 pickled
    object files with names given as parameters
    """
    if not options.input_file:
        p4a_util.die("Missing --input-file")

    if not options.output_file:
        p4a_util.die("Missing --output-file")

    if not os.path.exists(options.input_file):
        p4a_util.die("Input file does not exist: " + options.input_file)

    process_file(options.input_file, options.output_file)


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
