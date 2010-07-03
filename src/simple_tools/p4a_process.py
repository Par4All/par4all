#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
# - Ronan Keryell <ronan.keryell@hpc-project.com>
#

'''
Par4All processing
'''

import sys, os, re, shutil
from p4a_util import *


default_properties = dict(
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
    # Warning: assume that there is no aliasing between IO
    # streams ('FILE *' variables):
    ALIASING_ACROSS_IO_STREAMS = False,
    # Warning: this is a work in progress. Assume no weird
    # aliasing
    CONSTANT_PATH_EFFECTS = False,
    # Prevents automatic addition of OpenMP directives when
    # unslitting.  We will add them manually using ompify if
    # requested.
    PRETTYPRINT_SEQUENTIAL_STYLE = "do"
)

# Import of pyps will be done manually.
# Module instance will be held in following variable.
pyps = None


class p4a_processor_output():
    files = []
    database_dir = ""
    exception = None


class p4a_processor_input():
    "Store options given to the process engine, mainly digested by PyPS"
    project_name = ""
    accel = False
    cuda = False
    openmp = False
    fine = False
    select_modules = ""
    exclude_modules = ""
    cpp_flags = ""
    files = []
    recover_includes = True
    properties = {}


def add_module_options(parser):
    "Add to an optparse option parser the options related to this module"
    group = optparse.OptionGroup(parser, "Processor Options")

    group.add_option("--input-file", metavar = "FILE", default = None,
        help = "Input file (as created using the pickle module on a p4a_processor_input instance).")

    group.add_option("--output-file", metavar = "FILE", default = None,
        help = "Output file (to be created using the pickle module on a p4a_processor_output instance).")

    parser.add_option_group(group)


def process(input):

    output = p4a_processor_output()

    try:
        # Create a workspace with PIPS:
        processor = p4a_processor(
            project_name = input.project_name,
            cpp_flags = input.cpp_flags,
            verbose = True,
            files = input.files,
            filter_select = input.select_modules,
            filter_exclude = input.exclude_modules,
            accel = input.accel,
            cuda = input.cuda,
            recover_includes = input.recover_includes,
            properties = input.properties,
            # Regions are a must! :-) Ask for most precise regions:
            activates = [ "MUST_REGIONS" ]
        )

        output.database_dir = processor.get_database_directory()

        # First apply some generic parallelization:
        processor.parallelize(input.fine)

        if input.cuda:
            processor.gpuify()

        if input.openmp:
            processor.ompify()

        # Write the output files.
        output.files = processor.save()

    except:
        e = sys.exc_info()[1]
        if e.__class__.__name__ == "RuntimeError" and str(e).find("pips") != -1:
            output.exception = p4a_error("An error occurred in PIPS while processing " + ", ".join(input.files))
        else:
            #~ error("Processing of " + ", ".join(input.files) + " failed")
            output.exception = e

    return output


def process_file(input_file, output_file):
    input = load_pickle(input_file)
    output = process(input)
    save_pickle(output_file, output)


def main(options, args = []):

    if not options.input_file:
        die("Missing --input-file")

    if not options.output_file:
        die("Missing --output-file")

    if not os.path.exists(options.input_file):
        die("Input file does not exist: " + options.input_file)

    process_file(options.input_file, options.output_file)


class p4a_processor():
    """Process program files with PIPS and other tools"""

    # If the main language is Fortran, set to True:
    fortran = None

    workspace = None

    main_filter = None

    # The project name:
    project_name = None

    # Set to True to try to do some #include tracking and recovering
    recover_includes = None

    files = []
    accel_files = []

    def __init__(self, workspace = None, project_name = "", cpp_flags = "",
                 verbose = False, files = [], filter_select = None,
                 filter_exclude = None, accel = False, cuda = False,
                 recover_includes = True, properties = {}, activates = []):

        self.recover_includes = recover_includes
        self.accel = accel
        self.cuda = cuda

        if workspace:
            self.workspace = workspace
        else:
            # This is because pyps.workspace.__init__ will test for empty
            # strings
            if cpp_flags is None:
                cpp_flags = ""

            if not project_name:
                raise p4a_error("Missing project_name")

            self.project_name = project_name

            if self.recover_includes:
                # Use a special preprocessor to track #include by a
                # man-in-the-middle attack :-) :
                os.environ['PIPS_CPP'] = 'p4a_recover_includes --simple -E'

            for file in files:
                if self.fortran is None:
                    # Track the language for an eventual later compilation
                    # by a back-end target compiler. The first file type
                    # select the type for all the workspace:
                    if fortran_file_p(file):
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

            # Late import of pyps to avoid importing until
            # we really need it.
            global pyps
            try:
                pyps = __import__("pyps")
            except:
                raise

            # Create the PyPS workspace.
            self.workspace = pyps.workspace(self.files,
                                            name = self.project_name,
                                            activates = activates,
                                            verboseon = verbose,
                                            cppflags = cpp_flags)

            global default_properties
            all_properties = default_properties
            for k in properties:
                all_properties[k] = properties[k]
            for k in all_properties:
                debug("Property " + k + " = " + str(all_properties[k]))
            self.workspace.set_property(**all_properties)

        # Skip the compilation units and the modules of P4A runtime, they
        # are just here so that PIPS has a global view of what is going
        # on, not to be parallelized :-)
        skip_p4a_runtime_and_compilation_unit_re = re.compile("P4A_.*|.*!")

        # Also filter out modules based on --include-modules and
        # --exclude-modules.
        filter_select_re = None
        if filter_select:
            filter_select_re = re.compile(filter_select)
        filter_exclude_re = None
        if filter_exclude:
            filter_exclude_re = re.compile(filter_exclude)
        # Combine the 3 filters in one:
        self.main_filter = (lambda module: not skip_p4a_runtime_and_compilation_unit_re.match(module.name)
            and (filter_exclude_re == None or not filter_exclude_re.match(module.name))
            and (filter_select_re == None or filter_select_re.match(module.name)))


    def get_database_directory(self):
        "Return the directory of the current PIPS database"
        return os.path.abspath(self.workspace.directory())


    def filter_modules(self, filter_select = None, filter_exclude = None, other_filter = lambda x: True):
        """Filter modules according to their names and select them if they
        match all the 3 following conditions.

        If filter_exclude regex if defined, then matching modules are
        filtered out.

        If filter_select regex if defined, the matching modules are kept.

        If other_filter regex if defined, select also according to this
        matching.

        """
        filter_select_re = None
        if filter_select:
            filter_select_re = re.compile(filter_select)

        filter_exclude_re = None
        if filter_exclude:
            filter_exclude_re = re.compile(filter_exclude)

        filter = (lambda module: self.main_filter(module)
            and (filter_exclude_re == None or not filter_exclude_re.match(module.name))
            and (filter_select_re == None or filter_select_re.match(module.name))
            and other_filter(module.name))
        # Select the interesting modules:
        return self.workspace.filter(filter)


    def parallelize(self, fine = False, filter_select = None, filter_exclude = None):
        all_modules = self.filter_modules(filter_select, filter_exclude)

        # Try to privatize all the scalar variables in loops:
        all_modules.privatize_module()

        if fine:
            # Use a fine-grain parallelization à la Allen & Kennedy:
            all_modules.internalize_parallel_code()
        else:
            # Use a coarse-grain parallelization with regions:
            all_modules.coarse_grain_parallelization()


    def gpuify(self, filter_select = None, filter_exclude = None):
        all_modules = self.filter_modules(filter_select, filter_exclude)

        # In CUDA there is a limitation on 2D grids of thread blocks, in
        # OpenCL there is a 3D limitation, so limit parallelism at 2D
        # top-level loops inside parallel loop nests:
        all_modules.limit_nested_parallelism(NESTED_PARALLELISM_THRESHOLD = 2)

        # First, only generate the launchers to work on them later. They are
        # generated by outlining parallel loops
        all_modules.gpu_ify(GPU_USE_WRAPPER = False,
                            GPU_USE_KERNEL = False)

        # Select kernel launchers by using the fact that all the generated
        # functions have their name beginning with "p4a_kernel_launcher":
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

        # Select kernels by using the fact that all the generated kernels
        # have their names of this form:
        kernel_filter_re = re.compile("p4a_kernel_\\d+$")
        kernels = self.workspace.filter(lambda m: kernel_filter_re.match(m.name))

        # Unfortunately CUDA 3.0 does not accept C99 array declarations
        # with sizes also passed as parameters in kernels. So we degrade
        # the quality of the generated code by generating array
        # declarations as pointers and by accessing them as
        # array[linearized expression]:
        kernels.array_to_pointer(ARRAY_TO_POINTER_FLATTEN_ONLY = True,
                                 ARRAY_TO_POINTER_CONVERT_PARAMETERS= "POINTER")

        # Indeed, it is not only in kernels but also in all the CUDA code
        # that these C99 declarations are forbidden. We need them in the
        # original code for more precise analysis but we need to remove
        # them everywhere :-(
        #self.workspace.all_functions.array_to_pointer(
        #    ARRAY_TO_POINTER_FLATTEN_ONLY = True,
        #    ARRAY_TO_POINTER_CONVERT_PARAMETERS = "POINTER"
        #    )

        #self.workspace.all_functions.display()

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


    def ompify(self, filter_select = None, filter_exclude = None):
        """Add OpenMP #pragma from internal representation"""

        modules = self.filter_modules(filter_select, filter_exclude);
        modules.ompify_code()
        modules.omp_merge_pragma()


    def accel_post(self, file, dest_dir = None):
        '''Method for post processing "accelerated" files'''

        info("Post-processing " + file)

        post_process_script = os.path.join(get_program_dir(), "p4a_post_processor.py")

        args = [ post_process_script ]
        if dest_dir:
            args += [ '--dest-dir', dest_dir ]
        args.append(file)

        run(args)
        #~ subprocess.call(args)


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


    def __del__(self):
        # Waiting for pyps.workspace.close!
        if self.workspace:
            del self.workspace


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
