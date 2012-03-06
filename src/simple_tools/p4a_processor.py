#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
# - Ronan Keryell <ronan.keryell@hpc-project.com>
# + Many others...

# Beware: class p4a_scmp_compiler declared in ../scmp/p4a_scmp_compiler.py
# inherits from class p4a_processor.
# Maybe a common parent class with the minimal set of shared features
# should be defined, from which all compilers (say p4a_cuda_compiler,
# p4a_openmp_compiler) would inherit. BC.

import p4a_util
import optparse
import subprocess
import sys
import os
import re
import shutil
import pypsex

'''
Par4All processing
'''


# Basic properties to be used in Par4All:
default_properties = dict(
    # Useless to go on if something goes wrong... :-(
    #ABORT_ON_USER_ERROR = True,
    ABORT_ON_USER_ERROR = False,

    # Compute the intraprocedural preconditions at the same
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

    # Prevents automatic pretty-printing of OpenMP directives when
    # unsplitting.  We will add them using ompify if requested.
    PRETTYPRINT_SEQUENTIAL_STYLE = "do",

    # Required property since floating point arithmetic operations are not
    # associative because of rounding. PIPS does not take that into
    # account now and is based on theoretical math... cf PIPS TRAC #551

    # Well, François Irigoin seems to have improved this, so avoid
    # the spam of parenthesis...
    # PRETTYPRINT_ALL_PARENTHESES = True
)

# The default values of some PIPS properties are OK for C but has to be
# redefined for FORTRAN
default_fortran_cuda_properties = dict(
    GPU_KERNEL_PREFIX                     = "P4A_KERNEL",
    GPU_WRAPPER_PREFIX                    = "P4A_WRAPPER",
    GPU_LAUNCHER_PREFIX                   = "P4A_LAUNCHER",
    GPU_FORTRAN_WRAPPER_PREFIX            = "P4A_F08_WRAPPER",
    CROUGH_SCALAR_BY_VALUE_IN_FCT_DECL    = True,
    CROUGH_SCALAR_BY_VALUE_IN_FCT_CALL    = True,
    PRETTYPRINT_STATEMENT_NUMBER          = False,
    CROUGH_FORTRAN_USES_INTERFACE         = True,
    KERNEL_LOAD_STORE_LOAD_FUNCTION       = "P4A_COPY_TO_ACCEL",
    KERNEL_LOAD_STORE_LOAD_FUNCTION_1D    = "P4A_COPY_TO_ACCEL_1D",
    KERNEL_LOAD_STORE_LOAD_FUNCTION_2D    = "P4A_COPY_TO_ACCEL_2D",
    KERNEL_LOAD_STORE_LOAD_FUNCTION_3D    = "P4A_COPY_TO_ACCEL_3D",
    KERNEL_LOAD_STORE_LOAD_FUNCTION_4D    = "P4A_COPY_TO_ACCEL_4D",
    KERNEL_LOAD_STORE_LOAD_FUNCTION_5D    = "P4A_COPY_TO_ACCEL_5D",
    KERNEL_LOAD_STORE_LOAD_FUNCTION_6D    = "P4A_COPY_TO_ACCEL_6D",
    KERNEL_LOAD_STORE_ALLOCATE_FUNCTION   = "P4A_ACCEL_MALLOC",
    KERNEL_LOAD_STORE_STORE_FUNCTION      = "P4A_COPY_FROM_ACCEL",
    KERNEL_LOAD_STORE_STORE_FUNCTION_1D   = "P4A_COPY_FROM_ACCEL_1D",
    KERNEL_LOAD_STORE_STORE_FUNCTION_2D   = "P4A_COPY_FROM_ACCEL_2D",
    KERNEL_LOAD_STORE_STORE_FUNCTION_3D   = "P4A_COPY_FROM_ACCEL_3D",
    KERNEL_LOAD_STORE_STORE_FUNCTION_4D   = "P4A_COPY_FROM_ACCEL_4D",
    KERNEL_LOAD_STORE_STORE_FUNCTION_5D   = "P4A_COPY_FROM_ACCEL_5D",
    KERNEL_LOAD_STORE_STORE_FUNCTION_6D   = "P4A_COPY_FROM_ACCEL_6D",
    KERNEL_LOAD_STORE_DEALLOCATE_FUNCTION = "P4A_ACCEL_FREE",
    KERNEL_LOAD_STORE_VAR_SUFFIX          = "_num"
)

# Import of pyps will be done manually.
# Module instance will be held in following variable.
pyps = None

def apply_user_requested_phases(modules=None, phases_to_apply=[]):
    """Apply user requested phases to modules
    """
    for ph in phases_to_apply:
        # Apply requested phases to modules:
        getattr(modules, ph)()

class p4a_processor_output(object):
    files = []
    database_dir = ""
    exception = None


class p4a_processor_input(object):
    """Store options given to the process engine, mainly digested by PyPS.
    Some of the options are used during the output file generation.
    """
    project_name = ""
    noalias = False
    pointer_analysis = False
    accel = False
    cuda = False
    opencl = False
    com_optimization = False
    cuda_cc = 2
    fftw3 = False
    openmp = False
    scmp = False
    fine_grain = False
    c99 = False
    select_modules = ""
    exclude_modules = ""
    cpp_flags = ""
    files = []
    recover_includes = True
    native_recover_includes = False
    properties = {}
    output_dir=None
    output_suffix=""
    output_prefix=""
    brokers=""
    # To store some arbitrary Python code to be executed inside p4a_process:
    execute_some_python_code_in_process = None
    apply_phases={}


class p4a_processor(object):
    """Process program files with PIPS and other tools
    """
    # If the main language is Fortran, set to True:
    fortran = None

    workspace = None

    main_filter = None

    # The project name:
    project_name = None

    # Set to True to try to do some #include tracking and recovering
    recover_includes = None
    native_recover_includes = None

    # Initialize some lists of files:
    # - the list of the input files
    files = []
    # - the list of the p4a stub files
    accel_files = []

    # The list of the "on the fly" resources generated by PIPS:
    # - generated modules (by PIPS):
    generated_modules = []
    # - the list of the crough modules:
    crough_modules = []
    # - the list of module with interfaces:
    interface_modules = []
    # - the generated header files:
    header_files = []
    # - the set of CUDA modules:
    cuda_modules = set ()
    # - the set of C modules:
    c_modules = set ()
    # - the list of Fortran modules:
    fortran_modules = set ()

    # - the list of kernel names:
    kernels = []
    # - the list of launcher names:
    launchers = []

    # Some constants to be used for the PIPS generated files:
    new_files_folder  = "p4a_new_files"
    new_files_include = new_files_folder + "_include.h"

    # The typedef to be used in CUDA to flag kernels
    kernel_return_type  = "P4A_accel_kernel"
    wrapper_return_type = "P4A_accel_kernel_wrapper"


    def __init__(self, workspace = None, project_name = "", cpp_flags = "",
                 verbose = False, files = [], filter_select = None,
                 filter_exclude = None, noalias = False, pointer_analysis = False, 
                 accel = False, cuda = False, opencl = False, openmp = False, 
                 com_optimization = False, cuda_cc=2, fftw3 = False,
                 recover_includes = True, native_recover_includes = False,
                 c99 = False, use_pocc = False, pocc_options = "", atomic = False, brokers="",
                 properties = {}, apply_phases={}, activates = []):

        self.noalias = noalias
        self.pointer_analysis = pointer_analysis
        self.recover_includes = recover_includes
        self.native_recover_includes = native_recover_includes
        self.accel = accel
        self.cuda = cuda
        self.opencl = opencl
        self.openmp = openmp
        self.com_optimization = com_optimization
        self.cuda_cc = cuda_cc
        self.fftw3 = fftw3
        self.c99 = c99
        self.pocc = use_pocc
        self.pocc_options = pocc_options
        self.atomic = atomic
        self.apply_phases = apply_phases

        if workspace:
            # There is one provided: use it!
            self.workspace = workspace
        else:
            # This is because pyps.workspace.__init__ will test for empty
            # strings...
            if cpp_flags is None:
                cpp_flags = ""

            if not project_name:
                raise p4a_util.p4a_error("Missing project_name")

            self.project_name = project_name

            # The generated kernels source files will go into a directory named
            # with project_name.generated
            self.new_files_folder = self.project_name + '.generated'

            if self.recover_includes and not self.native_recover_includes:
                # Use a special preprocessor to track #include by a
                # man-in-the-middle attack :-) :
                os.environ['PIPS_CPP'] = 'p4a_recover_includes --simple -E'

            for file in files:
                if self.fortran is None:
                    # Track the language for an eventual later compilation
                    # by a back-end target compiler. The first file type
                    # select the type for all the workspace:
                    if p4a_util.fortran_file_p(file):
                        self.fortran = True
                    else:
                        self.fortran = False

                if not os.path.exists(file):
                    raise p4a_util.p4a_error("File does not exist: " + file)

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

            # Late import of pyps to avoid importing it until
            # we really need it.
            global pyps
            global broker
            global pocc
            try:
                pyps = __import__("pyps")
                broker = __import__("broker")
                if self.pocc:
                    pocc = __import__("pocc")
            except:
                raise

            # If we have #include recovery and want to use the native one:
            recover_Include = self.recover_includes and self.native_recover_includes
            # Create the PyPS workspace:
            if brokers != "":
                brokers+=","
            brokers+="p4a_stubs_broker"
            self.workspace = broker.workspace(*self.files,
                                              name = self.project_name,
                                              verbose = verbose,
                                              cppflags = cpp_flags,
                                              recoverInclude = recover_Include,
                                              brokersList=brokers)

            # Array regions are a must! :-) Ask for most precise array
            # regions:
            self.workspace.activate("MUST_REGIONS")

            if self.noalias:
                # currently, as default  PIPS phases do not select pointer analysis, setting
                # properties is sufficient.
                # activating phases may become necessary if the default behavior
                # changes in Pips
                properties["CONSTANT_PATH_EFFECTS"] = False
                properties["TRUST_CONSTANT_PATH_EFFECTS_IN_CONFLICTS"] = True
                
            if pointer_analysis:
                properties["ABSTRACT_HEAP_LOCATIONS"]="context-sensitive"
                self.workspace.activate("proper_effects_with_points_to")
                self.workspace.activate("cumulated_effects_with_points_to")
                self.workspace.activate("must_regions_with_points_to")

            # set the workspace properties
            self.set_properties(properties)


        # Skip the compilation units and the modules of P4A runtime, they
        # are just here so that PIPS has a global view of what is going
        # on, not to be parallelized :-)
        skip_p4a_runtime_and_compilation_unit_re = re.compile("P4A_.*|.*!$")

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
        

    def set_properties (self, user_properties):
        """ Initialize the properties according to the default defined properties
        and to the user defined ones.
        """
        global default_properties
        global default_fortran_cuda_properties
        all_properties = default_properties
        # if accel (might be cuda) and fortran add some properties
        if ((self.accel == True) and (self.fortran == True)):
            for k in default_fortran_cuda_properties:
                all_properties[k] = default_fortran_cuda_properties[k]
        # overwrite default properties with the user defined ones
        for k in user_properties:
            all_properties[k] = user_properties[k]
        # apply the properties to the workspace
        for k in all_properties:
            p4a_util.debug("Property " + k + " = " + str(all_properties[k]))
            setattr(self.workspace.props,k, all_properties[k])
        return

    def get_database_directory(self):
        "Return the directory of the current PIPS database"
        return os.path.abspath(self.workspace.dirname)


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


    # RK: I think the following should be in another file because it
    # clutters the global compilation sketch

    def post_process_fortran_wrapper (self, file_name, subroutine_name):
        """ All the dirty thing about C and Fortran interoperability is hidden
        in one unique file, the Fortran wrapper. This method does the last
        modification to make the file compilable by gfortran.

        Those steps are done:
        1 - insert the needed "use" statement
        2 - insert the pointer declaration for P4A inserted variable
        3 - insert the 64 bit integer var
        3 - types are written with a () in the size_of function
        4 - remove the f77 multiline. Also remove the first blank chars
        5 - replace f77 comments with f95 comments
        6 - remove the (void **) & that is not useful in fortran
        7 - remove the * in front of the inserted variables
        8 - declare the origin_var_name as a target
        9 - make DMA transfer variable to be 64 bit variables
        """
        p4a_util.debug ("Processing fortran_wrapper " + file_name)
        indent = "  "
        # Get the code to be post process:
        code = p4a_util.read_file (file_name, True)

        # Step 1

        # Insert the needed use statement right after the subroutine
        # declaration: common interface to be used:
        use_string  = indent + "use iso_c_binding\n"
        use_string += indent + "use p4a_runtime_interface\n"
        # Add the dedicated interface i.e use the KERNEL_LAUNCHER prefix
        # instead of the FORTRAN_WRAPPER prefix:
        use_string += indent + "use "
        use_string += subroutine_name.replace(self.get_fortran_wrapper_prefix(),
                                              self.get_launcher_prefix())

        use_string += "_interface\n"

        # Step 2

        # First identify the inserted variable:
        var_prefix = self.get_kernel_load_store_var_prefix ()
        var_suffix = self.get_kernel_load_store_var_suffix ()
        var_re = re.compile(var_prefix + "\\w+" + var_suffix + "\\d+")
        inserted_var_l = var_re.findall (code)
        origin_var_s = set ()
        inserted_var_s = set ()
        p4a_util.add_list_to_set (inserted_var_l, inserted_var_s)
        inserted_var_decl = indent + "type (c_ptr) ::"
        first = True
        for var in inserted_var_s:
            if (first == True):
                first = False
            else:
                inserted_var_decl += ","
            inserted_var_decl += " " + var
            # Extract the original variable name:
            origin_var_s.add (var.replace (var_prefix, "").replace (var_suffix, ""))
        inserted_var_decl += "\n" + indent + "integer (c_size_t), target :: p4a_zero"

        # Step 3
        c_sizeof_replace = dict()
        c_sizeof_replace ["CHARACTER()"] = "c_char"
        c_sizeof_replace ["LOGICAL()"]   = "c_bool"
        c_sizeof_replace ["INTEGER()"]   = "c_int"
        c_sizeof_replace ["INTEGER*4()"] = "c_int"
        c_sizeof_replace ["INTEGER*8()"] = "c_long_long"
        c_sizeof_replace ["REAL()"]      = "c_float"
        c_sizeof_replace ["REAL*4()"]    = "c_float"
        c_sizeof_replace ["REAL*8()"]    = "c_double"
        for k, v in c_sizeof_replace.iteritems ():
            code = code.replace (k, v)

        # Step 4
        F77_INDENTATION  = "\n      "
        F95_INDENTATION  = "\n" + indent
        F77_CONTINUATION = "\n     &"
        F95_CONTINUATION = "&\n      "
        code = code.replace (F77_CONTINUATION, " ")
        code = code.replace (F77_INDENTATION, F95_INDENTATION)

        # Step 5
        F77_CONTINUATION = "\nC"
        F95_CONTINUATION = "\n!"
        code = code.replace (F77_CONTINUATION, F95_CONTINUATION)

        # Step 6
        code = code.replace ("(void **) &", "")

        # Step 7
        for var in inserted_var_s:
            code = code.replace ("*" + var, var)

        # Step 8
        # Insert the target attribute for all declared variables:
        types_l = ["CHARACTER", "LOGICAL", "INTEGER*4", "INTEGER*8", "INTEGER",
                   "REAL*4", "REAL*8","REAL"]
        for t in types_l:
            code = code.replace (t, t.lower () + ", target ::")

        # Step 9
        function_l = ["P4A_COPY_FROM_ACCEL_2D", "P4A_COPY_TO_ACCEL_2D"]
        for func in function_l:
            # This RE matches the full line where the CALL to the DMA
            # transfer happens:
            func_line_re = re.compile("^ *CALL " + func + "\(.*\)",
                                           re.MULTILINE)
            # This RE matches the same line and extract the parameters:
            func_sig_re = re.compile("^ *CALL " + func + "\((.*)\)",
                                          re.MULTILINE)
            # This RE match the function name:
            func_name_re = re.compile("^ *(CALL " + func + ")\(.*\)",
                                           re.MULTILINE)
            new_line_l = []
            line_l = func_line_re.findall (code)
            name_l = func_name_re.findall (code)
            sig_l = func_sig_re.findall (code)
            insert_init = True
            for index in range (len (line_l)):
                # For each match we need to ensure that parameter 2..7 are
                # 64 bit long:
                no_space = sig_l[index].replace (" ", "")
                arg_l = no_space.split (",")
                for arg_num in range (1,7):
                    arg_l[arg_num] += "+p4a_zero"
                    new_line = indent + name_l [index] +"("
                    first = True
                for arg in arg_l:
                    if first:
                        first = False
                    else:
                        new_line += ","
                    new_line += arg
                new_line += ")"
                if insert_init:
                    insert_init = False
                    code = code.replace (line_l[index], indent + "p4a_zero = 0\n" + line_l[index])
                code = code.replace (line_l[index], new_line)

        # Step 10
        # Identify where to insert the USE string and the inserted variable
        # declaration:
        subroutine_line_re = re.compile("SUBROUTINE " + subroutine_name + ".*$",
                                        re.MULTILINE)
        subroutine_l = subroutine_line_re.findall (code)
        assert (len (subroutine_l) == 1)
        code = code.replace (subroutine_l[0], subroutine_l[0] + "\n" +
                             use_string + inserted_var_decl)

        # Write the post processed code:
        p4a_util.write_file(file_name, code, True)

        return

    def generated_modules_is_empty (self):
        return (len (self.generated_modules) == 0)

    def crough_modules_is_empty (self):
        return (len (self.crough_modules) == 0)

    def interface_modules_is_empty (self):
        return (len (self.interface_modules) == 0)

    def get_launcher_prefix (self):
        return self.workspace.props.GPU_LAUNCHER_PREFIX

    def get_kernel_prefix (self):
        return self.workspace.props.GPU_KERNEL_PREFIX

    def get_wrapper_prefix (self):
        return self.workspace.props.GPU_WRAPPER_PREFIX

    def get_fortran_wrapper_prefix (self):
        return self.workspace.props.GPU_FORTRAN_WRAPPER_PREFIX

    def get_kernel_load_store_var_prefix (self):
        return self.workspace.props.KERNEL_LOAD_STORE_VAR_PREFIX

    def get_kernel_load_store_var_suffix (self):
        return self.workspace.props.KERNEL_LOAD_STORE_VAR_SUFFIX

    def fortran_wrapper_p (self, file_name):
        prefix = self.get_fortran_wrapper_prefix()
        fortran_wrapper_file_name_re = re.compile(prefix + "_\\w+.f[0-9]*")
        m = fortran_wrapper_file_name_re.match (os.path.basename (file_name))
        return (m != None)

    def parallelize(self, fine_grain = False, filter_select = None,
                    filter_exclude = None, apply_phases_before = [], apply_phases_after = [], omp=False):
        """Apply transformations to parallelize the code in the workspace
        """
        all_modules = self.filter_modules(filter_select, filter_exclude)
        

        if fine_grain:
            # Set to False (mandatory) for A&K algorithm on C source file
            self.workspace.props.memory_effects_only = self.fortran

        # Apply requested phases before parallezation
        apply_user_requested_phases(all_modules, apply_phases_before)

        # Try to privatize all the scalar variables in loops:
        all_modules.privatize_module()

        # Use a different //izing scheme for openmp and the other accelerators
        # Wait for p4a 2.0 for better engineering
        if omp:
            # first step is to find big parallel loops
            all_modules.coarse_grain_parallelization(concurrent=True)
            # and the one with reductions
            all_modules.flag_parallel_reduced_loops_with_openmp_directives(concurrent=True)
            # on the **others**, try to distribute them
            if fine_grain:
                self.workspace.props.parallelize_again_parallel_code=False
                self.workspace.props.memory_effects_only = False # mandatory for internalize_parallel_code
                all_modules.internalize_parallel_code(concurrent=True)
                # and flag the remaining reductions if possible 
                # !! Show first a test case where it is useful !!
                # all_modules.flag_parallel_reduced_loops_with_openmp_directives(concurrent=True)
        else:
            if fine_grain:
                # Use a fine-grain parallelization à la Allen & Kennedy:
                all_modules.internalize_parallel_code(concurrent=True)

            # Always use a coarse-grain parallelization with regions:
            all_modules.coarse_grain_parallelization(concurrent=True)


        #all_modules.flatten_code(unroll=False,concurrent=True)
        #all_modules.simplify_control(concurrent=True)
        all_modules.loop_fusion(concurrent=True)
        #all_modules.localize_declaration(concurrent=True)

        # Scalarization doesn't preserve perfect loop nest at that time
        #all_modules.scalarization(concurrent=True)

        # Privatization information has been lost because of flatten_code
        #all_modules.privatize_module()
        #if fine_grain:
            # Use a fine-grain parallelization à la Allen & Kennedy:
            #all_modules.internalize_parallel_code(concurrent=True)


        # Apply requested phases after parallelization:
        apply_user_requested_phases(all_modules, apply_phases_after)


    def gpuify(self, filter_select = None,
                filter_exclude = None,
                fine_grain = False,
                apply_phases_kernel = [],
                apply_phases_kernel_launcher = [],
                apply_phases_wrapper = [],
                apply_phases_after = []):
        """Apply transformations to the parallel loop nested found in the
        workspace to generate GPU-oriented code
        """
        all_modules = self.filter_modules(filter_select, filter_exclude)

        # Some "risky" optimizations
        #all_modules.flatten_code(unroll=False,concurrent=True)
        #all_modules.simplify_control(concurrent=True)
        #all_modules.loop_fusion(concurrent=True)
        # Have to debug (see polybench/2mm.c)
        #all_modules.localize_declaration(concurrent=True)
        #all_modules.scalarization(concurrent=True)

        # We handle atomic operations here
        if self.atomic:
            # Idem for this phase:
            all_modules.replace_reduction_with_atomic(concurrent=True)

        # In CUDA there is a limitation on 2D grids of thread blocks, in
        # OpenCL there is a 3D limitation, so limit parallelism at 2D
        # top-level loops inside parallel loop nests:
        # Fermi and more recent device allows a 3D grid :)
        if self.cuda_cc >= 2 :
            all_modules.limit_nested_parallelism(NESTED_PARALLELISM_THRESHOLD = 3, concurrent=True)
        else:
            all_modules.limit_nested_parallelism(NESTED_PARALLELISM_THRESHOLD = 2, concurrent=True)


        # First, only generate the launchers to work on them later. They
        # are generated by outlining all the parallel loops. In the
        # Fortran case, we want the launcher to be wrapped in an
        # independent Fortran function so that it will be prettyprinted
        # later in... C (for OpenCL or CUDA kernel definition). :-)

        # go through the call graph in a top - down fashion
        def gpuify_all(module):
                module.gpu_ify(GPU_USE_WRAPPER = False, 
                            GPU_USE_KERNEL = False,                             
                            GPU_USE_FORTRAN_WRAPPER = self.fortran,
                            GPU_USE_LAUNCHER = True,
                            GPU_USE_LAUNCHER_INDEPENDENT_COMPILATION_UNIT = self.c99,
                            GPU_USE_KERNEL_INDEPENDENT_COMPILATION_UNIT = self.c99,
                            GPU_USE_WRAPPER_INDEPENDENT_COMPILATION_UNIT = self.c99,
                            OUTLINE_WRITTEN_SCALAR_BY_REFERENCE = False, # unsure
                            OUTLINE_CALLEES_PREFIX="p4a_device_",
                            annotate_loop_nests = True) # annotate for recover parallel loops later
                # recursive walk through
                [gpuify_all(c) for c in module.callees if c.name.find(self.get_launcher_prefix ()) !=0 and c not in all_modules]

        # call gpuify_all recursively starting from the heads of the callgraph
        # Keep in mind that all_modules can be filtered !!!
        [ gpuify_all(m) for m in all_modules if not [val for val in all_modules if val in m.callers]]



        # Select kernel launchers by using the fact that all the generated
        # functions have their names beginning with the launcher prefix:
        launcher_prefix = self.get_launcher_prefix ()
        kernel_launcher_filter_re = re.compile(launcher_prefix + "_.*[^!]$")
        kernel_launchers = self.workspace.filter(lambda m: kernel_launcher_filter_re.match(m.name) and not m.static_p())

        # We flag loops in kernel launchers as parallel, based on the annotation
        # previously made
        kernel_launchers.gpu_parallelize_annotated_loop_nest();
        kernel_launchers.gpu_clear_annotations_on_loop_nest();

        # Normalize all loops in kernels to suit hardware iteration spaces:
        kernel_launchers.loop_normalize(
            # Loop normalize to be GPU friendly, even if the step is already 1:
            LOOP_NORMALIZE_ONE_INCREMENT = True,
            # Arrays start at 0 in C, 1 in Fortran so the iteration loops:
            LOOP_NORMALIZE_LOWER_BOUND = 1 if self.fortran else 0,
            # It is legal in the following by construction (...hmmm to verify)
            LOOP_NORMALIZE_SKIP_INDEX_SIDE_EFFECT = True,
            concurrent=True)

        # Apply requested phases to kernel_launchers:
        apply_user_requested_phases(kernel_launchers, apply_phases_kernel_launcher)

        # End to generate the wrappers and kernel contents, but not the
        # launchers that have already been generated:
        kernel_launchers.gpu_ify(GPU_USE_LAUNCHER = False,
								 # opencl option will produce independent kernel and wrapper files
								 GPU_USE_KERNEL_INDEPENDENT_COMPILATION_UNIT = self.opencl,
								 GPU_USE_WRAPPER_INDEPENDENT_COMPILATION_UNIT = False,
								 OUTLINE_INDEPENDENT_COMPILATION_UNIT = self.c99,
								 OUTLINE_WRITTEN_SCALAR_BY_REFERENCE = False, # unsure
								 concurrent=True)

        # Select kernels by using the fact that all the generated kernels
        # have their names of this form:
        kernel_prefix = self.get_kernel_prefix ()
        kernel_filter_re = re.compile(kernel_prefix + "_\\w+$")
        kernels = self.workspace.filter(lambda m: kernel_filter_re.match(m.name))

        # scalarization is a nice optimization :)
        # currently it's very limited when applied in kernel, but cannot be applied outside neither ! :-(
        kernels.scalarization(concurrent=True)

		# Apply requested phases to kernel:
        apply_user_requested_phases(kernels, apply_phases_kernel)

        # Select wrappers by using the fact that all the generated wrappers
        # have their names of this form:
        wrapper_prefix = self.get_wrapper_prefix()
        wrapper_filter_re = re.compile(wrapper_prefix  + "_\\w+$")
        wrappers = self.workspace.filter(lambda m: wrapper_filter_re.match(m.name))

        # clean all, this avoid lot of warnings at compile time
        all_modules.clean_declarations()
        kernels.clean_declarations()
        wrappers.clean_declarations()
        kernel_launchers.clean_declarations()


        if not self.com_optimization :
            # Add communication around all the call site of the kernels. Since
            # the code has been outlined, any non local effect is no longer an
            # issue:
            kernel_launchers.kernel_load_store(concurrent=True,
                                               ISOLATE_STATEMENT_EVEN_NON_LOCAL = True
                                               )
        else :
            # The following should be done somewhere else with a generic
            # stub concept... When it is available.

            # Identify kernels first
            kernel_launchers.flag_kernel()
            # Kernels for fftw3 runtime:
            fftw3_kernel_filter_re = re.compile("^fftw.?_execute")
            fftw3_kernels = self.workspace.filter(lambda m: fftw3_kernel_filter_re.match(m.name))
            fftw3_kernels.flag_kernel()
            self.workspace.fun.main.kernel_data_mapping(KERNEL_LOAD_STORE_LOAD_FUNCTION="P4A_runtime_copy_to_accel",KERNEL_LOAD_STORE_STORE_FUNCTION="P4A_runtime_copy_from_accel")

		# Apply requested phases to wrappers:
        apply_user_requested_phases(wrappers, apply_phases_wrapper)

        # Wrap kernel launch for communication optimization runtime:
        if self.com_optimization :
            wrappers.wrap_kernel_argument(WRAP_KERNEL_ARGUMENT_FUNCTION_NAME="P4A_runtime_host_ptr_to_accel_ptr")
            wrappers.cast_at_call_sites()


        # Select Fortran wrappers by using the fact that all the generated
        # Fortran wrappers have their names of this form:
        f_wrapper_prefix = self.get_fortran_wrapper_prefix ()
        f_wrapper_filter_re = re.compile(f_wrapper_prefix  + "_\\w+$")
        f_wrappers = self.workspace.filter(lambda m: f_wrapper_filter_re.match(m.name))



        # Unfortunately CUDA (at least up to 4.0) does not accept C99
        # array declarations with sizes also passed as parameters in
        # kernels. So, we degrade the quality of the generated code by
        # generating array declarations as pointers and by accessing them
        # as array[linearized expression]:
        if self.c99 or self.fortran or self.opencl:
            skip_static_length_arrays = self.c99 and not self.opencl
            use_pointer = self.c99 or self.opencl
            kernel_launchers.linearize_array(use_pointers=use_pointer,cast_at_call_site=True,skip_static_length_arrays=skip_static_length_arrays)
            wrappers.linearize_array(use_pointers=use_pointer,cast_at_call_site=True,skip_static_length_arrays=skip_static_length_arrays)

            def linearize_all(k):
                k.linearize_array(use_pointers=use_pointer,cast_at_call_site=True,skip_static_length_arrays=skip_static_length_arrays, skip_local_arrays=True) # always skip locally declared arrays for kernels. Assume there is no VLA in the kernel, which woul elad to an alloca anyway
                [ linearize_all(c) for c in k.callees ]
            [ linearize_all(c) for c in kernels ]

        # SG: not usefull anymore. Uncomment this if you want to try it again, this is the right place to do it
        ## Unfold kernel, usually won't hurt code size, but less painful with
        ## static functions declared in accelerator compilation units 
        #kernels.unfold()

        # add sentinel around loop nests in launcher, used to replace the loop
        # nest with a call kernel in post-processing
        kernel_launchers.gpu_loop_nest_annotate(parallel=True);

        # Update the list of CUDA modules:
        p4a_util.add_list_to_set (map(lambda x:x.name, kernels),
                                  self.cuda_modules)
        p4a_util.add_list_to_set (map(lambda x:x.name, wrappers),
                                  self.cuda_modules)

        # Set return type for wrappers && kernel:
        if (self.fortran == False):
            wrappers.set_return_type_as_typedef(SET_RETURN_TYPE_AS_TYPEDEF_NEW_TYPE=self.wrapper_return_type)
            kernels.set_return_type_as_typedef(SET_RETURN_TYPE_AS_TYPEDEF_NEW_TYPE=self.kernel_return_type)
            if (self.c99 == True):
                self.generated_modules.extend (map(lambda x:x.name, kernel_launchers))
                #self.generated_modules.extend (map(lambda x:x.name, wrappers))
                #self.generated_modules.extend (map(lambda x:x.name, kernels))
        else:
            # RK: in the following, I don't understand why we display things...

            # Generate the C version of kernels, wrappers and launchers.
            # Kernels and wrappers need to be prettyprinted with arrays as
            # pointers because they will be .cu files
            kernels.display ("c_printed_file",
                             CROUGH_INCLUDE_FILE_LIST="p4a_accel.h",
                             DO_RETURN_TYPE_AS_TYPEDEF=True,
                             CROUGH_ARRAY_PARAMETER_AS_POINTER=True,
                             SET_RETURN_TYPE_AS_TYPEDEF_NEW_TYPE=self.kernel_return_type)
            wrappers.display ("c_printed_file",
                              CROUGH_INCLUDE_FILE_LIST="p4a_accel.h",
                              DO_RETURN_TYPE_AS_TYPEDEF=True,
                              CROUGH_ARRAY_PARAMETER_AS_POINTER=True,
                              SET_RETURN_TYPE_AS_TYPEDEF_NEW_TYPE=self.wrapper_return_type)
            # RK: following comment to be fixed...

            # Apply the set_return_type_as_typedef phase using regular
            # expressions because the phase is not available in Fortran
            # kernel_launchers will be .c file so C99 is allowed
            kernel_launchers.display ("c_printed_file",
                                      DO_RETURN_TYPE_AS_TYPEDEF=False,
                                      CROUGH_ARRAY_PARAMETER_AS_POINTER=False)

            # RK: should be done in 1 line...
            # Those newly generated modules have to be appended to the
            # dedicated list for later processing:
            self.crough_modules.extend (map(lambda x:x.name, kernels))
            self.crough_modules.extend (map(lambda x:x.name, wrappers))
            self.crough_modules.extend (map(lambda x:x.name, kernel_launchers))
            # Generate the interface of the wrappers. This will be used to call
            # the C functions of the wrappers from the fortran_wrapper
            # subroutines:
            kernel_launchers.print_interface ()
            self.interface_modules.extend (map(lambda x:x.name, kernel_launchers))
            self.generated_modules.extend (map(lambda x:x.name, f_wrappers))

		# Apply requested phases to kernels, wrappers and kernel_launchers
		# after gpuify():
        apply_user_requested_phases(kernels, apply_phases_after)
        apply_user_requested_phases(wrappers, apply_phases_after)
        apply_user_requested_phases(kernel_launchers, apply_phases_after)

        #self.workspace.all_functions.display()

        # Save the list of kernels for later work:
        self.kernels.extend (map(lambda x:x.name, kernels))
        if self.cuda:
			self.launchers.extend (map(lambda x:x.name, kernel_launchers))
        if self.opencl:
            # Comment the place where the opencl wrapper declaration must be placed
            # from the post-process
            for launcher in kernel_launchers:
                self.workspace[launcher.name].prepend_comment(PREPEND_COMMENT = "Opencl wrapper declaration\n")
            self.generated_modules.extend(map(lambda x:x.name, wrappers))



        # To be able to inject Par4All accelerator run time initialization
        # later:
        if "main" in self.workspace:
            self.workspace["main"].prepend_comment(PREPEND_COMMENT = "// Prepend here P4A_init_accel\n")
        else:
            p4a_util.warn('''
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


    def ompify(self,
            filter_select = None,
            filter_exclude = None,
            apply_phases_before = [],
            apply_phases_after = []):
        """Add OpenMP #pragma from loop-parallel flag internal
        representation to generate... OpenMP code!"""

        modules = self.filter_modules(filter_select, filter_exclude);

		# Apply requested phases before ompify to modules:
        apply_user_requested_phases(modules, apply_phases_before)
        modules.ompify_code(concurrent=True)
        modules.omp_merge_pragma(concurrent=True)

        if self.pocc:
            for m in modules:
                try:
                    m.poccify(options=self.pocc_options)
                except RuntimeError:
                    e = sys.exc_info()
                    p4a_util.warn("PoCC returned an error : " + str(e[1]))

		# Apply requested phases after ompify to modules:
        apply_user_requested_phases(modules, apply_phases_after)

    def accel_post(self, file, dest_dir = None):
        '''Method for post processing "accelerated" files'''

        p4a_util.info("Post-processing " + file)

        post_process_script = os.path.join(p4a_util.get_program_dir(), "p4a_post_processor.py")

        args = [ post_process_script ]
        if dest_dir:
            args += [ '--dest-dir', dest_dir ]
        args.append(file)

        p4a_util.run(args,force_locale = None)
        #~ subprocess.call(args)

    def get_p4a_accel_defines (self):
        defines = []
        defines.append ("-D")
        defines.append ("__thread=""")
        defines.append ("-DP4A_ACCEL_OPENMP")
        defines.append ("-D" + self.wrapper_return_type + "=void")
        defines.append ("-D" + self.kernel_return_type + "=void")
        return defines

    def kernel_to_wrapper_name (self, name):
        """ Return the wrapper name according to the kernel name using the
        good pips property.
        """
        return name.replace (self.get_kernel_prefix (), self.get_wrapper_prefix ())

    def kernel_to_launcher_name (self, name):
        """ Return the launcher name according to the kernel name using the
        good pips property.
        """
        return name.replace (self.get_kernel_prefix (), self.get_launcher_prefix ())

    def launchers_insert_extern_C (self):
        """Insert the extern C block construct to the whole file. The all
        the file functions will be callable from a C code.
        """
        for launcher in self.launchers:
            # Where the file does well in the .database workspace:
            launcher_file = os.path.join(self.workspace.dirname, "Src",
                                         launcher + ".c")
            # First open for read and get content:
            src = open (launcher_file, 'r')
            lines = src.readlines ()
            src.close ()
            # Then add the extern C block:
            dst = open (launcher_file, 'w')
            dst.write ('#ifdef __cplusplus\nextern "C" {\n#endif\n')
            for line in lines:
                dst.write (line)
            dst.write ("\n#ifdef __cplusplus\n}\n#endif\n")
            dst.close ()


    def merge_lwk (self):
        """ merge launcher wrapper and kernel in one file. The order is
        important the launcher call the wrapper that call the kernel. So
        they have to be in the inverse order into the file.
        """
        for kernel in self.kernels:
            # find the associated wrapper with the kernel
            wrapper  = self.kernel_to_wrapper_name  (kernel)
            launcher = self.kernel_to_launcher_name (kernel)
            # merge the files in the kernel file
            # Where the files do dwell in the .database workspace:

            wrapper_file = os.path.join(self.workspace.dirname, "Src",
                                       wrapper + ".c")
            kernel_file = os.path.join(self.workspace.dirname, "Src",
                                        kernel + ".c")
            launcher_file = os.path.join(self.workspace.dirname, "Src",
                                         launcher + ".c")

            if self.cuda:
				p4a_util.merge_files (kernel_file, [wrapper_file, launcher_file])
				# remove the wrapper from the modules to be processed since already
				#in the kernel
				self.generated_modules.remove (wrapper)
				self.generated_modules.remove (launcher)

    def save_header (self, output_dir, name):
        content = "/*All the generated includes are summarized here*/\n\n"
        for header in self.header_files:
            content += '#include "' + header + '"\n'
        p4a_util.write_file (os.path.join (output_dir, name), content)

    def save_crough (self, output_dir):
        """ Save the crough files that might have been generated by
        PIPS during the p4a process. Those files need a special handling since
        they are not produced in the standard Src folder by the unsplit phase.
        """
        result = []
        for name in self.crough_modules:
            # Where the file does well in the .database workspace:
            pips_file = os.path.join(self.workspace.dirname,
                                     name, name + ".c")
            # set the destination file
            output_name = name + ".c"
            if name in self.cuda_modules:
                if self.cuda:
                    output_name = p4a_util.change_file_ext(output_name, ".cu")
                # generate the header file
                header_file = os.path.join(output_dir, name + ".h")
                self.header_files.append (name + ".h")
                p4a_util.generate_c_header (pips_file, header_file,
                                            self.get_p4a_accel_defines ())
            # The final destination
            output_file = os.path.join(output_dir, output_name)
            # Copy the PIPS production to its destination:
            shutil.copyfile(pips_file, output_file)
            result.append(output_file)
        return result

    def save_generated (self, output_dir, subs_dir):
        """ Save the generated files that might have been generated by
        PIPS during the p4a process.
        """
        result = []
        if (self.fortran == True):
            extension_in = ".f"
            extension_out = ".f08"
        elif (self.opencl == True):
            extension_in = ".cl"
            extension_out = ".cl"
        else:
            extension_in = ".c"
            if (self.cuda == True):
                extension_out = ".cu"
#            elif (self.opencl == True):
                #extension_in = ".cl"
#                extension_out = ".cl"
            else:
                extension_out = ".c"
        #p4a_util.warn("generated modules length "+str(len(self.generated_modules)))

        for name in self.generated_modules:
            p4a_util.debug("Save generated : '" + name + "'")
            # Where the file actually is in the .database workspace:
            pips_file = os.path.join(self.workspace.dirname, "Src",
                                     name + extension_in)

            #p4a_util.warn("pips_file " +pips_file)
            if self.accel and (p4a_util.c_file_p(pips_file) or p4a_util.opencl_file_p(pips_file)):
                # We generate code for P4A Accel, so first post process
                # the output and produce the result in the P4A subdiretory
                # of the .database
                self.accel_post(pips_file,
                                os.path.join(self.workspace.dirname, "P4A"))
                # update the pips file to the postprocess one
                pips_file = os.path.join(self.workspace.dirname, "P4A", name + extension_in)

            output_name = name + extension_out
            # The final destination
            output_file = os.path.join(output_dir, output_name)
            if (self.fortran_wrapper_p (pips_file) == True):
                self.post_process_fortran_wrapper (pips_file, name)

            # Copy the PIPS production to its destination:
            shutil.copyfile(pips_file, output_file)
            result.append(output_file)

            if (self.opencl == True):
                # Merging the content of the p4a_accel_wrapper-OpenCL.h
                # in the .cl kernel file
                end_file =  os.path.join(subs_dir, output_name)
                #p4a_util.warn("end file "+end_file)
                # In the merge operation, the output file is only open in
                # append mode. When multiple compilation are launched,
                # without cleaning, the resulting file cumulates all
                # the versions. Removing the file before, prevent from this
                try:
                    os.remove(end_file)
                except os.error:
                    pass
                h_file = os.path.join(os.environ["P4A_ROOT"],"share","p4a_accel","p4a_accel_wrapper-OpenCL.h")
                p4a_util.merge_files (end_file, [h_file, output_file])
                #p4a_util.warn("end_file after join "+end_file)


            if (self.fortran == False):
                # for C generate the header file
                header_file = os.path.join(output_dir, name + ".h")
                self.header_files.append (name + ".h")
                p4a_util.generate_c_header (pips_file, header_file,
                                            self.get_p4a_accel_defines ())
        return result

    def save_interface (self, output_dir):
        """ Save the interface files that might have been generated during by
        PIPS during the p4a process. Those files need a special handling since
        they are not produced in the standard Src folder by the unsplit phase.
        """
        result = []
        flag = True
        for name in self.interface_modules:
            # Where the file does well in the .database workspace:
            pips_file = os.path.join(self.workspace.dirname, name,
                                     name + "_interface.f08")
            output_name = name + "_interface.f08"
            # The final destination
            output_file = os.path.join(output_dir, output_name)
            # Copy the PIPS production to its destination:
            shutil.copyfile(pips_file, output_file)
            result.append(output_file)
            if flag:
                result.append (os.path.join(os.environ["P4A_ACCEL_DIR"],
                                            "p4a_runtime_interface.f95"))
                flag = False
        return result

    def save_user_file (self, dest_dir, prefix, suffix):
        """ Save the user file appended to the Workspace at the begining
        """
        result = []
        # For all the user defined files from the workspace:
        for file in self.files:
            if file in self.accel_files:
                # We do not want to remove the stubs file from the
                # distribution... :-/
                #os.remove(file)
                continue
            (dir, name) = os.path.split(file)
            # Where the file does well in the .database workspace:
            pips_file = os.path.join(self.workspace.dirname, "Src", name)
            #p4a_util.warn("pips_file save_user_file "+pips_file)
            # Recover the includes in the given file only if the flags have
            # been previously set and this is a C program:
            if self.recover_includes and not self.native_recover_includes and p4a_util.c_file_p(file):
                subprocess.call([ 'p4a_recover_includes',
                                  '--simple', pips_file ])

            # Update the destination directory if one was given:
            if dest_dir:
                dir = dest_dir
                if not (os.path.isdir(dir)):
                    os.makedirs (dir)

            output_name = prefix + name
            if suffix:
                output_name = p4a_util.file_add_suffix(output_name, suffix)

            # The final destination
            output_file = os.path.join(dir, output_name)

            if self.accel and p4a_util.c_file_p(file):
                # We generate code for P4A Accel, so first post process
                # the output:

                self.accel_post(pips_file,
					os.path.join(self.workspace.dirname, "P4A"))
                # Where the P4A output file does dwell in the .database
                # workspace:
                p4a_file = os.path.join(self.workspace.dirname, "P4A", name)
                # Update the normal location then:
                pips_file = p4a_file
                #p4a_util.warn("pips_file save_user_file 2 "+pips_file)

                if (self.cuda == True) and (self.c99 == False):
                    # some C99 syntax is forbidden with Cuda. That's why there is
                    # a --c99 option that allows to generate a unique call site into the
                    # c99 original file to the wrappers (and then kenel). In such a case
                    # the original files will remain standard c99 files and the cuda files
                    # will only be the wrappers and the kernel (cf save_generated).
					output_file = p4a_util.change_file_ext(output_file, ".cu")
                #if (self.opencl == True):
                    #self.merge_function_launcher(pips_file)
                    #self.accel_post(pips_file)
                    #output_file = p4a_util.change_file_ext(output_file, ".c")

            # Copy the PIPS production to its destination:
            shutil.copyfile(pips_file, output_file)
            result.append (output_file)
        return result

    def save(self, dest_dir = None, prefix = "", suffix = "p4a"):
        """Final post-processing and save the files of the workspace. This
        includes the original files defined by the user and also all new
        files that might have been generated by PIPS, including headers.
        """

        output_files = []

        # Do not allow None suffix or prefix:
        if prefix is None:
            prefix = ""
        if suffix is None:
            suffix = ""

        # Set the suffix if needed to avoid file destruction
        if (dest_dir == None) and ( prefix == "") and (suffix == ""):
            suffix = "p4a"

        # Append or prepend the . to prefix or suffix
        if not (suffix == ""):
            suffix = "." + suffix
        if not (prefix == ""):
            prefix = prefix + "."

        # Regenerate the sources file in the workspace. Do not generate
        # OpenMP-style output since we have already added OpenMP
        # decorations:
        self.workspace.props.PRETTYPRINT_SEQUENTIAL_STYLE = "do"
        # The default place is fine for us since we work later on the files:
        self.workspace.save()

        # Create the folder for p4a new files if needed
        new_file_flag =  self.new_file_generated ()

        output_dir = os.path.join(os.getcwd(), self.new_files_folder)
        #p4a_util.warn("p4a new file " + output_dir)
        if dest_dir:
            output_dir = os.path.join(dest_dir,self.new_files_folder)
        if ((not (os.path.isdir(output_dir))) and (new_file_flag == True)):
            os.makedirs (output_dir)

        # For the opencl kernels that have been pushed in generated_files
        # but must be saved at the working place (substitutive directory)
        subs_dir = os.getcwd()
        if dest_dir:
            subs_dir = dest_dir

        # Nvcc compiles .cu files as C++, thus we add extern C { declaration
        # to prevent mangling
        if ((self.c99 == True) and (self.cuda == True)):
            self.launchers_insert_extern_C ()
            #no longer needed
            #self.merge_lwk ()


        # save the user files
        output_files.extend (self.save_user_file (dest_dir, prefix, suffix))

        if self.opencl:
            # HACK inside : we expect the wrapper and the kernel to be in the
            # same file which MUST be called wrapper_name.c
            for kernel in self.kernels:
                # find the associated wrapper with the kernel
                src_dir = os.path.join(self.workspace.dirname, "Src")
                wrapper  = os.path.join(src_dir,self.kernel_to_wrapper_name(kernel)+".cl")
                kernel = os.path.join(src_dir,kernel+".c")
                shutil.copyfile(kernel, wrapper)


        # save pips generated files in the dedicated folder
        output_files.extend (self.save_crough (output_dir))
        output_files.extend (self.save_interface (output_dir))
        output_files.extend (self.save_generated (output_dir, subs_dir))
        #output_files.extend (self.save_generated (output_dir))
        #p4a_util.warn("output_dir "+ output_dir)


        # generate one header to warp all the generated header files
        if (new_file_flag == True):
            self.save_header (output_dir, self.new_files_include)

        return output_files

    def new_file_generated (self):
        return not (self.generated_modules_is_empty () and
                    self.crough_modules_is_empty () and
                    self.interface_modules_is_empty ())

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
