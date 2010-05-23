#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Par4All Processing Class
'''

import sys, os, re, shutil
from p4a_util import *
import pyps

class p4a_processor():

    fortran = None
    workspace = None
    main_filter = None
    files = []
    accel_files = []

    def __init__(self, workspace = None, project_name = "", cppflags = "", verbose = False,
        files = [], filter_include = None, filter_exclude = None, accel = False):

        if workspace:
            self.workspace = workspace
        else:
            # This is because pyps.workspace.__init__ will test for empty strings
            if cppflags is None:
                cppflags = ""

            if not project_name:
                # Try some names until there is no database with this name:
                while True:
                    project_name = gen_name()
                    database_dir = os.path.join(os.getcwd(),
                                                project_name + ".database")
                    if not os.path.exists(database_dir):
                        break

            for file in files:
                if self.fortran is None:
                    (base, ext) = os.path.splitext(file)
                    if ext == ".f":
                        self.fortran = True
                    else:
                        self.fortran = False
                if not os.path.exists(file):
                    raise p4a_error("file does not exist: " + file)

            self.files = files

            if accel:
                # Why all that stuff ? Why not using them directly?
                # Because initially stubs.c/f were not removed because there was no post-processing.
                # And since the default behaviour of save() is to copy processed files to the original
                # source directory (with a prefix), this would have put the processed stubs in src/p4a_accel
                # for every project... So we copy the stubs to the current directory with a modified name
                # which includes the project name to prevent filename collision.
                accel_stubs_name = None
                if self.fortran:
                    accel_stubs_name = "p4a_stubs.f"
                else:
                    accel_stubs_name = "p4a_stubs.c"
                accel_stubs = os.path.join(os.environ["P4A_ACCEL_DIR"],
                                           accel_stubs_name)
                (base, ext) = os.path.splitext(os.path.basename(accel_stubs))
                output_accel_stubs = os.path.join(os.getcwd(), base + "_" + project_name + ext)
                debug("copying accel stubs: " + accel_stubs + " -> " + output_accel_stubs)
                shutil.copyfile(accel_stubs, output_accel_stubs)
                self.files += [ output_accel_stubs ]
                self.accel_files += [ output_accel_stubs ]

            # Use a special preprocessor to track #include:
            os.environ['PIPS_CPP'] = 'p4a_recover_includes --simple -E'

            # Create the PyPS workspace.
            self.workspace = pyps.workspace(self.files,
                                            name = project_name,
                                            activates = [],
                                            verboseon = verbose,
                                            cppflags = cppflags)
            self.workspace.w.set_property(
                    # Compute the intraprocedural preconditions at the
                    # same time as transformers and use them to improve
                    # the accuracy of expression and statement
                    # transformers:
                    SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT = True,
                    # Use the more precise fix point operator to cope with
                    # while loops:
                    SEMANTICS_FIX_POINT_OPERATOR = "derivative",
                    # Try to restructure the code for more precision:
                    UNSPAGHETTIFY_TEST_RESTRUCTURING = True,
                    UNSPAGHETTIFY_RECURSIVE_DECOMPOSITION = True,
                    # Simplify for loops into Fortran do-loops internally
                    # for better precision of analysis:
                    FOR_TO_DO_LOOP_IN_CONTROLIZER = True,
                    # Regions are a must! :-) Ask for most precise
                    # regions:
                    MUST_REGIONS = True,
                    # Warning: assume that there is no aliasing between IO
                    # streams ('FILE *' variables):
                    ALIASING_ACROSS_IO_STREAMS = False,
                    # Warning: this is a work in progress. Assume no weird
                    # aliasing
                    CONSTANT_PATH_EFFECTS = False
                    )

            # Useful? 
            #self.workspace.set_property(FOR_TO_DO_LOOP_IN_CONTROLIZER = True)

            # Prevents automatic addition of OpenMP directives when unsplitting.
            # We will add them manually using ompify if requested.
            self.workspace.set_property(PRETTYPRINT_SEQUENTIAL_STYLE = "do")

        # Useless to add redundant information:
        #for module in self.workspace:
           #module.prepend_comment(PREPEND_COMMENT = "/*\n * module " + module.name + "\n */\n")
                #+ " read on " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

        # Skip the compilation units and the modules of P4A runtime, they
        # are just here so that PIPS has a global view of what is going
        # on, not to be parallelized :-)
        skip_p4a_runtime_and_compilation_unit_re = re.compile("P4A_.*|.*!")

        # Also filter out modules based on --include-modules and
        # --exclude-modules.
        filter_include_re = None
        if filter_include:
            filter_include_re = re.compile(filter_include)
        filter_exclude_re = None
        if filter_exclude:
            filter_exclude_re = re.compile(filter_exclude)
        # Combine the 3 filters in one:
        self.main_filter = (lambda module: not skip_p4a_runtime_and_compilation_unit_re.match(module.name)
            and (filter_exclude_re == None or not filter_exclude_re.match(module.name))
            and (filter_include_re == None or filter_include_re.match(module.name)))


    def filter_modules(self, filter_include = None, filter_exclude = None, other_filter = lambda x: True):
        filter_include_re = None
        if filter_include:
            filter_include_re = re.compile(filter_include)
        filter_exclude_re = None
        if filter_exclude:
            filter_exclude_re = re.compile(filter_exclude)
        filter = (lambda module: self.main_filter(module)
            and (filter_exclude_re == None or not filter_exclude_re.match(module.name))
            and (filter_include_re == None or filter_include_re.match(module.name))
            and other_filter(module.name))
        # Select the interesting modules:
        return self.workspace.filter(filter)


    def parallelize(self, fine = False, filter_include = None, filter_exclude = None):
        all_modules = self.filter_modules(filter_include, filter_exclude)

        # Try to privatize all the scalar variables in loops:
        all_modules.privatize_module()

        if fine:
            # Use a fine-grain parallelization à la Allen & Kennedy:
            all_modules.internalize_parallel_code()
        else:
            # Use a coarse-grain parallelization with regions:
            all_modules.coarse_grain_parallelization()


    def gpuify(self, filter_include = None, filter_exclude = None):
        all_modules = self.filter_modules(filter_include, filter_exclude)

        # Normalize all loops:
        all_modules.loop_normalize(
            # Loop normalize for the C language and GPU friendly
            LOOP_NORMALIZE_ONE_INCREMENT = True,
            # Arrays start at 0 in C, so the iteration loops:
            LOOP_NORMALIZE_LOWER_BOUND = 0,
            # It is legal in the following by construction (...Hmmm to verify)
            LOOP_NORMALIZE_SKIP_INDEX_SIDE_EFFECT = True)

        all_modules.parallelize()

        # First, only generate the launchers to work on them later. They are
        # generated by outlining parallel loops
        all_modules.gpu_ify(GPU_USE_WRAPPER = False,
                            GPU_USE_KERNEL = False)

        # Isolate kernels by using the fact that all the generated kernels
        # have their name beginning with "p4a_":
        kernel_launcher_filter_re = re.compile("p4a_kernel_launcher_.*[^!]$")
        kernel_launchers = w.filter(lambda m: kernel_launcher_filter_re.match(m.name))

        # Add iteration space decoration and insert iteration clamping
        # into the launchers:
        kernel_launchers.gpu_loop_nest_annotate()

        # Some verification output:
        #kernel_launchers.display()

        # Unfortunately the information about parallelization and
        # privatization is lost by the current outliner, so rebuild
        # it... :-(
        kernel_launchers.privatize_module()
        kernel_launchers.coarse_grain_parallelization()

        # End to generate the wrappers and kernel contents, but not the
        # launchers that have already been generated:
        kernel_launchers.gpu_ify(GPU_USE_LAUNCHER = False)

        # Some verification output:
        #w.all_functions.display()
        #w.compilation_units.display()

        # Add communication around all the call site of the kernels:
        kernel_launchers.kernel_load_store()
        #kernel_launchers.display()

        # To be able to inject Par4All accelerator run time initializationlater:
        self.workspace["main"].prepend_comment(PREPEND_COMMENT = "// Prepend here P4A_init_accel")

    def ompify(self, filter_include = None, filter_exclude = None):
        self.filter_modules(filter_include, filter_exclude).ompify_code()


    def accel_post(self, file, dest_file = None):
        '''Method for post processing "accelerated" files'''

        info("post-processing " + file)

        content = slurp(file)

        ###
        # To catch stuff like
        # void p4a_kernel_launcher_1(void *accel_address, void *host_address, size_t n);
        kernel_launcher_declaration_re = re.compile("void (p4a_kernel_launcher_\\d+)[^;]+")
        # A mappping kernel_launcher_name -> declaration
        kernel_launcher_declarations = {}
        result = kernel_launcher_declaration_re.search(content)
        if result:
            kernel_launcher_declarations[result.group(1)] = result.group(0)
        debug("kernel launcher declarations: " + repr(kernel_launcher_declarations))
        ###

        global_sub_count = 0

        (content, sub_count) = re.subn("(?s)(/\\*\n \\* file for [^\n]+\n \\*/\n).*/\* Define some macros helping to catch buffer overflows.  \*/",
            "\\1#include <p4a_accel.h>\n#include <stdio.h>\n#include <math.h>\n", content)
        debug("clean-up headers and inject standard header injection: " + str(sub_count))
        global_sub_count += sub_count

        (content, sub_count) = re.subn("// Prepend here P4A_init_accel\n",
            "P4A_init_accel;\n", content)
        debug("compatibility: " + str(sub_count))
        global_sub_count += sub_count

        (content, sub_count) = re.subn(", (p4a_kernel[^0-9]+[0-9]+\\()",
                ";\n   void \\1", content)
        debug("put only one function per line for further replacement: " + str(sub_count))
        global_sub_count += sub_count

        (content, sub_count) = re.subn("(void p4a_kernel_wrapper_[0-9]+[^\n]+)",
                "P4A_accel_kernel_wrapper \\1", content)
        debug("add accelerator attributes on accelerated parts: " + str(sub_count))
        global_sub_count += sub_count

        (content, sub_count) = re.subn("(void p4a_kernel_[0-9]+[^\n]+)",
                "P4A_accel_kernel \\1", content)
        debug("add accelerator attributes on accelerated parts: " + str(sub_count))
        global_sub_count += sub_count

        (content, sub_count) = re.subn("(?s)// Loop nest P4A begin,(\\d+)D\\(([^)]+)\\).*?// Loop nest P4A end\n.*?(p4a_kernel_wrapper_\\d+)\\(([^)]*)\\);",
                "P4A_call_accel_kernel_\\1d(\\3, \\2, \\4);", content)
        debug("generate accelerated kernel calls: " + str(sub_count))
        global_sub_count += sub_count

        (content, sub_count) = re.subn("( *)// To be assigned to a call to (P4A_vp_[0-9]+): ([^\n]+)",
                 "\\1// Index has been replaced by \\2:\n\\1\\3 = \\2;", content)
        debug("get the virtual processor coordinates: " + str(sub_count))
        global_sub_count += sub_count

        info(str(global_sub_count) + " substitutions made")

        if not dest_file:
            dest_file = file
        debug("writing to " + dest_file)

        dump(dest_file, content)

        return global_sub_count


    def save(self, in_dir = None, prefix = "p4a_"):
        output_files = []
        self.workspace.all.unsplit()
        for file in self.files:
            if file in self.accel_files:
                os.remove(file)
                continue
            (dir, name) = os.path.split(file)
            pips_file = os.path.join(self.workspace.directory(), "Src", name)
            if in_dir:
                dir = in_dir
            if name[0:len(prefix)] != prefix:
                name = prefix + name
            output_file = os.path.join(dir, name)
            copy = True
            if len(self.accel_files):
                sub_count = self.accel_post(pips_file, output_file)
                if sub_count and not self.fortran:
                    cu_file = change_file_ext(output_file, ".cu")
                    shutil.move(output_file, cu_file)
                    output_file = cu_file
                    copy = False
            if copy:
                shutil.copyfile(pips_file, output_file)
            output_files += [ output_file ]
        return output_files


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
