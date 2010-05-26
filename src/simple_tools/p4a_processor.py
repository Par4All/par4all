#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
# - Ronan Keryell
#

'''
Par4All Processing Class
'''

import sys, os, re, shutil
from p4a_util import *
import pyps

class p4a_processor():
    """Process program files with PIPS and other tools"""

    # If the main language is Fortran, set to True:
    fortran = None

    workspace = None

    main_filter = None

    # The project name:
    project_name = None

    # The full name of the directory that store the workspace database:
    database_dir = None

    # Set to True to try to do some #include tracking and recovering
    recover_includes = None

    files = []
    accel_files = []

    def __init__(self, workspace = None, project_name = "", cppflags = "",
                 verbose = False, files = [], filter_include = None,
                 filter_exclude = None, accel = False, cuda = False,
                 recover_includes = True):

        self.recover_includes = recover_includes
        self.accel = accel
        self.cuda = cuda

        if workspace:
            self.workspace = workspace
        else:
            # This is because pyps.workspace.__init__ will test for empty
            # strings
            if cppflags is None:
                cppflags = ""

            if self.recover_includes:
                # Use a special preprocessor to track #include by a
                # man-in-the-middle attack :-) :
                os.environ['PIPS_CPP'] = 'p4a_recover_includes --simple -E'

            if not project_name:
                # Try some names until there is no database with this name:
                while True:
                    self.project_name = gen_name()
                    self.database_dir = os.path.join(os.getcwd(),
                                                self.project_name + ".database")
                    if not os.path.exists(self.database_dir):
                        break
            else:
                self.project_name = project_name

            for file in files:
                if self.fortran is None:
                    (base, ext) = os.path.splitext(file)
                    # Track the language for an eventual later compilation
                    # by a back-end target compiler. The first file type
                    # select the type for all the workspace:
                    if ext == ".f":
                        self.fortran = True
                    else:
                        self.fortran = False

                if not os.path.exists(file):
                    raise p4a_error("File does not exist: " + file)

            self.files = files

            if accel:
                accel_stubs_name = None
                # Analyze this stub file so that PIPS interprocedurality
                # is happy about the run-time we use:
                if self.fortran:
                    accel_stubs_name = "p4a_stubs.f"
                else:
                    accel_stubs_name = "p4a_stubs.c"
                # The stubs are here in our distribution:
                accel_stubs = os.path.join(os.environ["P4A_ACCEL_DIR"],
                                           accel_stubs_name)
                # Add the stubs file to the list to use in PIPS:
                self.files += [ accel_stubs ]
                # Mark this file as a stub to avoid copying it out later:
                self.accel_files += [ accel_stubs ]

            # Use a special preprocessor to track #include:
            os.environ['PIPS_CPP'] = 'p4a_recover_includes --simple -E'

            # Create the PyPS workspace.
            self.workspace = pyps.workspace(self.files,
                                            name = self.project_name,
                                            activates = [],
                                            verboseon = verbose,
                                            cppflags = cppflags)
            self.workspace.set_property(
                # Useless to go on if something goes wrong... :-(
                ABORT_ON_USER_ERROR = True,
                # Compute the intraprocedural preconditions at the same
                # time as transformers and use them to improve the
                # accuracy of expression and statement transformers:
                SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT = True,
                # Use the more precise fix point operator to cope with
                # while loops:
                SEMANTICS_FIX_POINT_OPERATOR = "derivative",
                # Try to restructure the code for more precision:
                UNSPAGHETTIFY_TEST_RESTRUCTURING = True,
                UNSPAGHETTIFY_RECURSIVE_DECOMPOSITION = True,
                # Simplify for loops into Fortran do-loops internally for
                # better precision of analysis:
                FOR_TO_DO_LOOP_IN_CONTROLIZER = True,
                # Regions are a must! :-) Ask for most precise regions:
                MUST_REGIONS = True,
                # Warning: assume that there is no aliasing between IO
                # streams ('FILE *' variables):
                ALIASING_ACROSS_IO_STREAMS = False,
                # Warning: this is a work in progress. Assume no weird
                # aliasing
                CONSTANT_PATH_EFFECTS = False,
                # Prevents automatic addition of OpenMP directives when
                # unslitting.  We will add them manually using ompify if
                # requested.
                PRETTYPRINT_SEQUENTIAL_STYLE = "do")

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

        # First, only generate the launchers to work on them later. They are
        # generated by outlining parallel loops
        all_modules.gpu_ify(GPU_USE_WRAPPER = False,
                            GPU_USE_KERNEL = False)

        # Isolate kernels by using the fact that all the generated kernels have
        # their name beginning with "p4a_":
        kernel_launcher_filter_re = re.compile("p4a_kernel_launcher_.*[^!]$")
        kernel_launchers = self.workspace.filter(lambda m: kernel_launcher_filter_re.match(m.name))

        # Normalize all loops in kernels to suit hardware iteration spaces:
        kernel_launchers.loop_normalize(
            # Loop normalize for the C language and GPU friendly
            LOOP_NORMALIZE_ONE_INCREMENT = True,
            # Arrays start at 0 in C, so the iteration loops:
            LOOP_NORMALIZE_LOWER_BOUND = 0,
            # It is legal in the following by construction (...Hmmm to verify)
            LOOP_NORMALIZE_SKIP_INDEX_SIDE_EFFECT = True)

        # Unfortunately the information about parallelization and
        # privatization is lost by the current outliner, so rebuild
        # it... :-( But anyway, since we've normalized the code, we
        # changed it so it is to be parallelized again...
        #kernel_launchers.privatize_module()
        kernel_launchers.capply("privatize_module")
        #kernel_launchers.coarse_grain_parallelization()
        kernel_launchers.capply("coarse_grain_parallelization")
        #kernel_launchers.localize_declaration()
        # Does not work:
        #kernel_launchers.omp_merge_pragma()


        # Add iteration space decoration and insert iteration clamping
        # into the launchers onto the outer parallel loop nests:
        kernel_launchers.gpu_loop_nest_annotate()

        # End to generate the wrappers and kernel contents, but not the
        # launchers that have already been generated:
        kernel_launchers.gpu_ify(GPU_USE_LAUNCHER = False)

        # Add communication around all the call site of the kernels:
        kernel_launchers.kernel_load_store()
        # To be able to inject Par4All accelerator run time initialization
        # later:
        if "main" in self.workspace:
            self.workspace["main"].prepend_comment(PREPEND_COMMENT = "// Prepend here P4A_init_accel")
        else:
            warn('''
            There is no "main()" function in the given sources.
            That means the P4A Accel runtime initialization can not be
            inserted and that the compiled application may not work.

            If you build a P4A executable from partial p4a output, you
            should add a
               #include <p4a_accel.h>
            at the beginning of the .c file containing the main()
            and add at the beginning of main() a line with:
               P4A_init_accel;
            ''')
    def ompify(self, filter_include = None, filter_exclude = None):
        """Add OpenMP #pragma from internal representation"""

        self.filter_modules(filter_include, filter_exclude).ompify_code()


    def accel_post(self, file, dest_dir = None):
        '''Method for post processing "accelerated" files'''

        info("Post-processing " + file)

        args = [ 'p4a_post_processor.py' ]

        if dest_dir:
            args += [ '--dest-dir', dest_dir ]

        args.append(file)

        subprocess.call(args)


    def save(self, in_dir = None, prefix = "", suffix = ".p4a"):
        """Final post-processing and save the files of the workspace"""

        output_files = []

        # Regenerate the sources file in the workspace. Do not generate
        # OpenMP-style output since we have already added OpenMP
        # decorations:
        self.workspace.all.unsplit(PRETTYPRINT_SEQUENTIAL_STYLE = "do")

        # For all the registered files from the workspace:
        for file in self.files:
            if file in self.accel_files:
                # We do not want to remove the stubs file from the
                # distribution... :-/
                #os.remove(file)
                continue
            (dir, name) = os.path.split(file)
            # Where the file does dwell in the .database workspace:
            pips_file = os.path.join(self.workspace.directory(), "Src", name)

            # Recover the includes in the given file only if the flag has
            # been previously set and this is a C program. Do not do it
            # twice in accel mode since it is already done in
            # p4a_post_processor.py:
            if self.recover_includes and c_file_p(file) and not self.accel:
                subprocess.call([ 'p4a_recover_includes',
                                  '--simple', pips_file ])

            # Update the destination directory if one was given:
            if in_dir:
                dir = in_dir

            if prefix is None:
                prefix = ""
            output_name = prefix + name

            if suffix:
                output_name = file_add_suffix(output_name, suffix)

            # The final destination
            output_file = os.path.join(dir, output_name)

            if self.accel and c_file_p(file):
                # We generate code for P4A Accel, so first post process
                # the output:
                self.accel_post(pips_file,
                                os.path.join(self.workspace.directory(), "P4A"))
                # Where the P4A output file does dwell in the .database
                # workspace:
                p4a_file = os.path.join(self.workspace.directory(), "P4A", name)
                # Update the normal location then:
                pips_file = p4a_file
                if self.cuda:
                    # To have the nVidia compiler to be happy, we need to
                    # have a .cu version of the .c file
                    output_file = change_file_ext(output_file, ".cu")

            # Copy the PIPS production to its destination:
            shutil.copyfile(pips_file, output_file)

            output_files.append(output_file)

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
