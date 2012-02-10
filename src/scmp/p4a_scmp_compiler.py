#!/usr/bin/env python 
# -*- coding: utf-8 -*-
#
# Author:
# -  Beatrice Creusillet <beatrice.creusillet@hpc-project.com>
#

import os
import shutil
import re
from p4a_processor import *
from p4a_util import *
import pyps

# The default values of some PIPS propeties are ok for C but have to be
# redefined for FORTRAN
default_scmp_properties = dict(
    GPU_KERNEL_PREFIX                     = "P4A_KERNEL",
    PRETTYPRINT_STATEMENT_NUMBER          = False,
    KERNEL_LOAD_STORE_LOAD_FUNCTION       = "P4A_copy_to_accel",
    KERNEL_LOAD_STORE_STORE_FUNCTION      = "P4A_copy_from_accel",
    KERNEL_LOAD_STORE_ALLOCATE_FUNCTION   = "P4A_accel_malloc",
    KERNEL_LOAD_STORE_DEALLOCATE_FUNCTION = "P4A_accel_free",
    KERNEL_LOAD_STORE_LOAD_FUNCTION_1D    = "P4A_copy_to_accel_1d",
    KERNEL_LOAD_STORE_STORE_FUNCTION_1D   = "P4A_copy_from_accel_1d",
    KERNEL_LOAD_STORE_LOAD_FUNCTION_2D    = "P4A_copy_to_accel_2d",
    KERNEL_LOAD_STORE_STORE_FUNCTION_2D   = "P4A_copy_from_accel_2d",
    KERNEL_LOAD_STORE_VAR_PREFIX          = "P4A_",
    KERNEL_LOAD_STORE_VAR_SUFFIX          = "__",
    ISOLATE_STATEMENT_VAR_PREFIX          = "P4A__",
    ISOLATE_STATEMENT_VAR_SUFFIX          = "__",
    ISOLATE_STATEMENT_EVEN_NON_LOCAL      = True,
    SCALOPES_KERNEL_TASK_PREFIX           = "scmp_task_"
)


class p4a_scmp_compiler(p4a_processor):

    scmp_applis_processing_dir_name = "applis_processing"
    scmp_control_task_dir_name = "applis"
    scmp_control_task_extension = ".app"
    p4a_scmp_header_file = "p4a_scmp.h"
    p4a_scmp_source_file = "p4a_scmp.c"
    scmp_prefix = "scmp_"
    scmp_default_stack_size = 1024
    scmp_buffers_file_name = "scmp_buffers.h"

    def __init__(self, files = [], project_name = "",scmp_applis_dir_name = ".",user_properties = {}, *sources, **kwargs): 
                 
        
        global default_scmp_properties
        self.scmp_applis_dir_name = scmp_applis_dir_name
        self.scmp_events_file_name = project_name + "_event_val.h"
        
        ## FIXME : what's the following ??? Stubs should be handled by a broker
        stubs_name = "p4a_stubs.c"
        self.stubs_files = os.path.join(os.environ["P4A_ACCEL_DIR"], stubs_name)
        files += [ self.stubs_files ]

        # add scmp properties to user properties
        properties = dict(user_properties)
        properties.update(default_scmp_properties)
        p4a_processor.__init__(self, files=files, project_name=project_name, properties=properties,  *sources, **kwargs)
        self.kernel_tasks_labels = []
        self.server_tasks_labels = []
        self.generated_files = []

    def save_user_files (self):
        """ Save the user files
        """
        print("Saving generated files to appli_processing directory\n")
        result = []
        # For all the user defined files from the workspace:
        for file in self.files:
            print("considering " + file + "... \n")
            if file in self.stubs_files:
                print ("stub file\n")
                continue
            (dir, name) = os.path.split(file)
            # Where the file actually is in the .database workspace:
            pips_file = os.path.join(self.workspace.dirname, "Src", name)

            # Recover the includes in the given file only if the flags have
            # been previously set and this is a C program:
            if self.recover_includes and not self.native_recover_includes and p4a_util.c_file_p(file):
                subprocess.call([ 'p4a_recover_includes',
                                  '--simple', pips_file ])

            # Update the destination directory if one was given:
            dir = os.path.join(self.scmp_applis_dir_name, p4a_scmp_compiler.scmp_applis_processing_dir_name, self.project_name)
            print("saving to directory: " + dir + "\n")
            if not (os.path.isdir(dir)):
                os.makedirs (dir)

            output_name = p4a_scmp_compiler.scmp_prefix + name
            print ("saving file: " + pips_file + " as " + output_name +"\n");

            # The final destination
            output_file = os.path.join(dir, output_name)

            # Copy the PIPS production to its destination:
            shutil.copyfile(pips_file, output_file)
            result.append (output_file)
            self.generated_files.append(output_file)

        print("done")
        return result

    def comment_out_old_stubs_declarations(self, files):
        """ comment out the olf stubs function declarations
        """
        print("Commenting out old stubs function declarations...")
        malloc_ch = "void " + default_scmp_properties["KERNEL_LOAD_STORE_ALLOCATE_FUNCTION"]
        free_ch = "void " + default_scmp_properties["KERNEL_LOAD_STORE_DEALLOCATE_FUNCTION"]
        copy_to_1d_ch = "void " + default_scmp_properties["KERNEL_LOAD_STORE_LOAD_FUNCTION_1D"]
        copy_from_1d_ch = "void " + default_scmp_properties["KERNEL_LOAD_STORE_STORE_FUNCTION_1D"]
        copy_to_2d_ch = "void " + default_scmp_properties["KERNEL_LOAD_STORE_LOAD_FUNCTION_2D"]
        copy_from_2d_ch = "void " + default_scmp_properties["KERNEL_LOAD_STORE_STORE_FUNCTION_2D"]
        for file in files:
            print ("considering file " + file + "...")
            ch=""
            with open(file, "r") as f:
                ch = f.read()
                ch = ch.replace(malloc_ch, "//" + malloc_ch)
                ch = ch.replace(free_ch, "//" + free_ch)
                ch = ch.replace(copy_to_1d_ch, "//" + copy_to_1d_ch)
                ch = ch.replace(copy_to_2d_ch, "//" + copy_to_2d_ch)
                ch = ch.replace(copy_from_1d_ch, "//" + copy_from_1d_ch)
                ch = ch.replace(copy_from_2d_ch, "//" + copy_from_2d_ch)

            with open(file, "w") as f:
                f.write(ch)
        print("done")

    def save_scmp_buffers_file(self):
        # Where the file is in the .database workspace:
        pips_file = os.path.join(self.workspace.dirname, "main", "main_buffers.h")

        # where we have to place it
        dir = os.path.join(self.scmp_applis_dir_name, p4a_scmp_compiler.scmp_applis_processing_dir_name, self.project_name)
        if not (os.path.isdir(dir)):
            os.makedirs (dir)
        output_file = os.path.join(dir, p4a_scmp_compiler.scmp_buffers_file_name)
        shutil.copyfile(pips_file, output_file)

    def recover_server_tasks(self):
        # Where the file is in the .database workspace:
        pips_file = os.path.join(self.workspace.dirname, "main", "main.servers")

        with open(pips_file, "r") as f:
            ch = f.read()
            self.server_tasks_labels = ch.split()

    def update_p4a_function_calls(self, files):
        # beware: regular expressions here are not really robust againts spacing changes.
        global default_scmp_properties
        print("updating calls to p4a_scmp_functions \n")
        malloc_re = re.compile(r"\(\(void \*\*\) \&([a-zA-Z0-9_]*), (sizeof\([ \w]*\)[ *\w]*)\);")
        copy_from_accel_re = re.compile(r"&(\w*)([a-zA-Z0-9_\[\]]*), \*(\w*)")
        copy_to_accel_re = re.compile(r"&(\w*)([a-zA-Z0-9_\[\]]*), \*(\w*)")
        dealloc_re = re.compile(r"\(([a-zA-Z0-9_]*)")
        for file in files:
            print ("considering file " + file + "...")
            os.rename(file, file+".tmp")
            with open(file+".tmp", "r") as f_orig:
                with open(file, "w") as f:
                    # headers first !
                    f.write("#include <sesam_com.h>\n")
                    f.write("#include \"" + p4a_scmp_compiler.p4a_scmp_header_file +"\" \n")
                    f.write("#include \"" + self.scmp_buffers_file_name + "\"\n")
                    f.write("#include \"" + self.scmp_events_file_name + "\"\n")
                    for line in f_orig:
                        if re.search(default_scmp_properties["KERNEL_LOAD_STORE_ALLOCATE_FUNCTION"], line) is not None:
                            other_line = malloc_re.sub(r"((void **) &\1, \2, \1_id, \1_prod_p || \1_cons_p, \1_prod_p);",
                                                       line)
                        elif re.search(default_scmp_properties["KERNEL_LOAD_STORE_DEALLOCATE_FUNCTION"], line) is not None:
                            other_line = dealloc_re.sub(r"(\1, \1_id, \1_prod_p || \1_cons_p, \1_prod_p", line)
                        elif re.search(default_scmp_properties["KERNEL_LOAD_STORE_STORE_FUNCTION_1D"], line) is not None:
                            other_line = copy_from_accel_re.sub(r"P4A_sesam_server_\1_p ? &\1\2 : NULL, *\3, \3_id, \3_prod_p || \3_cons_p", line)
                        elif  re.search(default_scmp_properties["KERNEL_LOAD_STORE_LOAD_FUNCTION_1D"], line) is not None:
                            other_line = copy_to_accel_re.sub(r"P4A_sesam_server_\1_p ? &\1\2 : NULL, *\3, \3_id, \3_prod_p || \3_cons_p", line)
                        elif re.search(default_scmp_properties["KERNEL_LOAD_STORE_STORE_FUNCTION_2D"], line) is not None:
                            other_line = copy_from_accel_re.sub(r"P4A_sesam_server_\1_p ? &\1\2 : NULL, *\3, \3_id, \3_prod_p || \3_cons_p", line)
                        elif  re.search(default_scmp_properties["KERNEL_LOAD_STORE_LOAD_FUNCTION_2D"], line) is not None:
                            other_line = copy_to_accel_re.sub(r"P4A_sesam_server_\1_p ? &\1\2 : NULL, *\3, \3_id, \3_prod_p || \3_cons_p", line)
                        else:
                            other_line = line
                        f.write(other_line)
            os.remove(file+".tmp")
            print ("done\n")

    def replace_kernel_tasks_labels(self, files):
        # replace kernel task labels by tests
        print("replacing kernel task labels with if construct\n")
        labels_re = re.compile(r"("+self.get_scalopes_kernel_task_prefix()+"\w+):")
        for file in files:
            print ("considering file " + file + "...")
            os.rename(file, file+".tmp")
            with open(file+".tmp", 'r') as f_orig:
                with open(file, 'w') as f:
                    for line in f_orig:
                        other_line = labels_re.sub(r"if (\1_p)",line)
                        f.write(other_line)
            os.remove(file+".tmp")
            print ("done\n")

    def specialize_task(self, original_task_name, file, task_name):
        print("replacing returns")
        main_re = re.compile(r"\bmain\(")
        return_re = re.compile(r"return[()\s\w]*;")
        bracket_re = re.compile("{")
        semicolon_re = re.compile(";")
        os.rename(file, file+".tmp")
        with open(file+".tmp", "r") as f_orig:
            with open(file, 'w') as f:
                f.write("#define " + original_task_name + " " + str(1) + "\n")
                found_main = None
                found_main_bracket = None
                found = False
                for line in f_orig:
                    other_line = line
                    if not found:
                        # find main
                        if found_main is None:
                            if semicolon_re.search(line) is None:
                                found_main = main_re.search(line)
                                found_main_bracket = bracket_re.search(line)
                                if found_main_bracket is not None:
                                    other_line = line + "\n\n\tP4A_scmp_reset();\n"
                        elif found_main_bracket is None:
                            found_main_bracket = bracket_re.search(line)
                            if found_main_bracket is not None:
                                other_line = line + "\n\n\tP4A_scmp_reset();\n"
                        else:
                            # find next return and replace by return(ev_T000x);
                            if return_re.search(line) is not None:
                                other_line = return_re.sub(r"return(ev_"+ task_name + ");\n", line)
                                found = True
                    f.write(other_line)
        os.remove(file+".tmp")
        print("done")

    def replace_server_copy_functions(self, file):
        """ Postfixes copy_to and copy_from function names
            by "_server"
        """
        global default_scmp_properties
        print("postfixing copy functions by _server in file " + file)
        copy_to_1d_ch = default_scmp_properties["KERNEL_LOAD_STORE_LOAD_FUNCTION_1D"]
        copy_from_1d_ch = default_scmp_properties["KERNEL_LOAD_STORE_STORE_FUNCTION_1D"]
        copy_to_2d_ch = default_scmp_properties["KERNEL_LOAD_STORE_LOAD_FUNCTION_2D"]
        copy_from_2d_ch = default_scmp_properties["KERNEL_LOAD_STORE_STORE_FUNCTION_2D"]
        os.rename(file, file+".tmp")
        with open(file+".tmp", 'r') as f_orig:
            ch = f_orig.read()
            with open(file, 'w') as f:
                ch = ch.replace(copy_to_1d_ch, copy_to_1d_ch + "_server")
                ch = ch.replace(copy_from_1d_ch, copy_from_1d_ch + "_server")
                ch = ch.replace(copy_to_2d_ch, copy_to_2d_ch + "_server")
                ch = ch.replace(copy_from_2d_ch, copy_from_2d_ch + "_server")
                f.write(ch)
        os.remove(file+".tmp")
        print ("done\n")

    def clone_tasks(self, files):
        """ Clone the file containing the main function for each
            kernel or server task
            main function return statement are replaced by
            return(ev_T00x); where x is the task number, and
            according to atomatic event header file generation
            by sesam
            In server tasks, copy_to and copy_from function names
            are postfixed by _server
        """
        # first look for the file in which there is the main function
        # because currently we only handle appliations in which tasks
        # all belong to the main function
        main_re = re.compile(r"\bmain\(")
        for file in files:
            with open(file, "r") as f:
                ch = f.read()
                if main_re.search(ch) is not None:
                    main_file = file
                else:
                    self.generated_files.append(file)
        print("file with main function is: " + main_file)

        nb_tasks = 0
        print("specializing main file code for kernel tasks\n")
        for kernel_task in self.kernel_tasks_labels:
            print("handling kernel task "+ kernel_task)
            nb_tasks += 1
            new_task_name = "T%03d" % (nb_tasks)
            new_task_file_name = new_task_name + ".mips.c"
            (dir, name) = os.path.split(main_file)
            new_task_file = os.path.join(dir, new_task_file_name)
            shutil.copyfile(file, new_task_file)

            # we then have to ad the #define kernel_task at the beginning
            # of the file and change the main return value
            self.specialize_task(kernel_task, new_task_file, new_task_name)
            print("generating " + new_task_name + "... done")
            self.generated_files.append(new_task_file)

        print("specializing code for server tasks\n")
        for server_task in self.server_tasks_labels:
            print("handling server task "+ server_task)
            nb_tasks += 1
            new_task_name = "T%03d" % (nb_tasks)
            new_task_file_name = new_task_name + ".mips.c"
            (dir, name) = os.path.split(main_file)
            new_task_file = os.path.join(dir, new_task_file_name)
            shutil.copyfile(file, new_task_file)

            # we then have to change copy functions names
            # and change the main return value
            self.specialize_task(server_task, new_task_file, new_task_name)
            self.replace_server_copy_functions(new_task_file)
            print("generating " + new_task_name + "... done")
            self.generated_files.append(new_task_file)

        print("Total number of tasks is : " + str(nb_tasks))
        return nb_tasks

    def generate_empty_task(self, task_number, event_string):
        """ Generation of task T00x as an empty task
            x being the provided task number
            the returned event is event_string
            Beware that the buffers header file is not included
        """
        print("Generating task number " + str(task_number))
        new_task_name = "T%03d" % (task_number)
        task_file_name = new_task_name + ".mips.c"
        dir = os.path.join(self.scmp_applis_dir_name, p4a_scmp_compiler.scmp_applis_processing_dir_name, self.project_name)
        if not (os.path.isdir(dir)):
            os.makedirs(dir)
        task_file = os.path.join(dir, task_file_name)
        with open(task_file, "w") as f:
            f.write("#include <sesam_com.h>\n")
            f.write("#include \""+ p4a_scmp_compiler.p4a_scmp_header_file + "\" \n")
            f.write("#include \"" + self.scmp_events_file_name + "\"\n")
            f.write("\nint main(){\n")
            f.write("\treturn("+ event_string + ");\n}\n")
            self.generated_files.append(task_file)
        print("done")

    def generate_control_task(self, nb_tasks):
        """ Generation of control task
        """
        print("Generating control task for " + str(nb_tasks) + " tasks")
        task_file_name = self.project_name + p4a_scmp_compiler.scmp_control_task_extension
        dir = os.path.join(self.scmp_applis_dir_name, p4a_scmp_compiler.scmp_control_task_dir_name)
        if not (os.path.isdir(dir)):
                os.makedirs(dir)
        task_file = os.path.join(dir, task_file_name)
        with open(task_file, "w") as f:
            f.write("NBTASK  " + str(nb_tasks) + "\n")
            f.write("NBEVENT " + str(nb_tasks-1) + "\n")
            f.write("DEADLINE	1000000\n")
            f.write("NAME_TASKS ")
            for task_number in range(nb_tasks):
                f.write(" T%03d" % (task_number))
            f.write("\nSOURCE   ")
            for task_number in range(nb_tasks):
                f.write(" T%03d" % (task_number))
            f.write("\nSIZE_STACK ")
            for task_number in range(nb_tasks):
                f.write(" " + str(p4a_scmp_compiler.scmp_default_stack_size))

            # declare T000 as initialization task, and declare
            # kernel and server tasks as children tasks
            f.write("\nINIT T000;")
            if nb_tasks > 1:
                f.write("\nNEXT T000 = ")
                for task_number in range(1, nb_tasks-1):
                    f.write("T%03d," % (task_number))
                f.write("ev_T000;")

            # declare finalization task as child of kernel and server tasks
            if nb_tasks > 1:
                f.write("\nNEXT ")
                for task_number in range(1, nb_tasks-2):
                    f.write("T%03d," % (task_number))
                f.write("T%03d = " % (nb_tasks-2))
                f.write("T%03d" % (nb_tasks-1))
                for task_number in range(1, nb_tasks-1):
                    f.write(",ev_T%03d" % (task_number))
                f.write(";")

            # declare finalization task
            f.write("\nNEXT T%03d = END;" % (nb_tasks-1))

            # tasks lengths
            f.write("\nLENGTH T000, 1 = 1000;")
            for task_number in range(1, nb_tasks-1):
                f.write("\nLENGTH T%03d, 1 = 10000;" % (task_number))
            f.write("\nLENGTH T%03d, 1 = 1000;" % (nb_tasks-1))

            # the end
            f.write("\nENDAPPLI;")
            self.generated_files.append(task_file)
        print("done")

    def export_makefile(self):
        """ Generate Makefile.arp in applis_processing directory
            It should add rules for all the files involved in the project
            Currently, it only copies the source Makefile.arp
        """
        print ("Exporting Makefile.arp...")
        source_dir = os.environ["P4A_SCMP_DIR"]
        makefile_name = "Makefile.arp"
        target_dir = os.path.join(self.scmp_applis_dir_name, p4a_scmp_compiler.scmp_applis_processing_dir_name, self.project_name)
        target_makefile = os.path.join(target_dir, makefile_name)
        shutil.copyfile(os.path.join(source_dir, makefile_name), target_makefile)
        self.generated_files.append(target_makefile)
        print("done")

    def export_p4a_scmp_files(self):
        """ Generate Makefile.arp in applis_processing directory
            It should add rules for all the files involved in the project
            Currently, it only copies the source Makefile.arp
        """
        print("Exporting p4a_scmp.h and p4a_scmp.c ...")

        source_dir = os.environ["P4A_SCMP_DIR"]
        target_dir = os.path.join(self.scmp_applis_dir_name, p4a_scmp_compiler.scmp_applis_processing_dir_name, self.project_name)

        header_name = p4a_scmp_compiler.p4a_scmp_header_file
        target_header_file = os.path.join(target_dir, header_name)

        source_name = p4a_scmp_compiler.p4a_scmp_source_file
        target_source_file = os.path.join(target_dir, source_name)

        shutil.copyfile(os.path.join(source_dir, header_name), target_header_file)
        self.generated_files.append(target_header_file)
        shutil.copyfile(os.path.join(source_dir, source_name), target_source_file)
        self.generated_files.append(target_source_file)
        print("done")

    def replace_printf(self, files):
        print("Replacing printf and fprintf(stdout, with sesam_printf...")
        printf_re = re.compile(r"\bprintf")
        fprintf_re = re.compile(r"\bfprintf\(stdout,")
        for file in files:
             os.rename(file, file+".tmp")
             with open(file+".tmp", 'r') as f_orig:
                 ch = f_orig.read()
                 with open(file, 'w') as f:
                     ch = printf_re.sub("sesam_printf", ch)
                     ch = fprintf_re.sub("sesam_printf(", ch)
                     f.write(ch)
             os.remove(file+".tmp")
        print("done")

    def get_scalopes_kernel_task_prefix (self):
        return self.workspace.props.scalopes_kernel_task_prefix

    def export_application_header_files(self):
        print("exporting user header files...")
        source_dir="."
        target_dir = os.path.join(self.scmp_applis_dir_name, p4a_scmp_compiler.scmp_applis_processing_dir_name, self.project_name)
        header_files = [file for file in os.listdir( source_dir) if get_file_extension(file) == '.h']

        for header_file in header_files:
            shutil.copyfile(os.path.join(source_dir, header_file), os.path.join(target_dir, header_file))
        print("done")

    def get_generated_files(self):
        return self.generated_files

    def go(self):
        global default_scmp_properties

        # currently only works for tasks declared in main()
        main = self.workspace.fun.main

        # recover a list of all task labels used in source code
        labels_re = re.compile(r"\n *("+self.get_scalopes_kernel_task_prefix()+"\w+):")
        self.kernel_tasks_labels = labels_re.findall(main.code)
        if self.kernel_tasks_labels:
            for kernel_task in self.kernel_tasks_labels:
                print("isolating kernel task "+ kernel_task + "...\n")
                main.isolate_statement(kernel_task)
                print("done \n")
            main.sesam_buffers_processing()
            #self.workspace.all.unsplit()
            self.workspace.save()

            # save the unsplitted file to the scmp applis_processing directory
            # and recover includes
            saved_files = self.save_user_files()

            # p4a_recover_includes does not deal with stubs files,
            # and we will modify function prototypes
            self.comment_out_old_stubs_declarations(saved_files)

            # copy header file in appli dir: only for main function for the moment
            self.save_scmp_buffers_file()

            # recover server tasks list
            self.recover_server_tasks()

            # update calls to the p4a_scmp functions and adding header file include
            self.update_p4a_function_calls(saved_files)

            # replace printf with sesam_printf in generated files
            self.replace_printf(saved_files)

            # replace kernel task labels by tests
            self.replace_kernel_tasks_labels(saved_files)

            # clone the application and specialize the clones for each task
            nb_tasks = self.clone_tasks(saved_files)

            # generate initialization and finalization tasks
            self.generate_empty_task(0, "ev_T000")
            self.generate_empty_task(nb_tasks+1, "END_APPLI")

            # generate Makefile.arp
            self.export_makefile()

            # generate control task
            self.generate_control_task(nb_tasks+2)

            # copy p4a_scmp files to applis_processing directory
            self.export_p4a_scmp_files()

            # copy original header files to the application directory
            # well this is now useless: p4a also recovers user header includes
            # still, I keep the line because it is useful
            # to export the whole application.
            self.export_application_header_files()

            # apply dead_code_elimination on each new application
            # beware to use proper stubs so that all code is not eliminated
            # not yet implemented


        else:
            print "no scmp task label found\n"

    def __del__(self):
        # Waiting for pyps.workspace.close!
        if self.workspace:
            del self.workspace


if __name__ == "__main__":
    print(__doc__)
    print("This module is not directly executable, use 'p4a --scmp' instead")

# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
