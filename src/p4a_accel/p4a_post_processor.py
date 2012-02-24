#! /usr/bin/env python

"""
Convert the PIPS C output to a P4A output

Ronan.Keryell@hpc-project.com
"""

#import string, re, sys, os, types, optparse

import sys, re, os, optparse, subprocess, p4a_util

verbose = False

# A mappping kernel_launcher_name -> declaration
kernel_launcher_declarations = {}

# To catch stuff like
# void p4a_kernel_launcher_1(void *accel_address, void *host_address, size_t n);
kernel_launcher_declaration = re.compile("void (p4a_kernel_launcher_\\w+)[^;]+")

def remove_libc_typedef (content):
    """ when outlining to independent compilation unit some typedef from the
    libc can be added here and there. This will make the compilation to fail.
    So remove them
    """
    size_t_re = re.compile ("typedef.*size_t");
    match_l = size_t_re.findall (content)
    for m in match_l:
        content = content.replace (m, "")
    return content

def gather_kernel_launcher_declarations(file_name):
    f = open(file_name)
    # slurp all the file in a string:
    content = f.read()
    f.close()
    result = kernel_launcher_declaration.search(content)
    if result:
        kernel_launcher_declarations[result.group(1)] = result.group(0)


def insert_kernel_launcher_declaration(m):
    "Insert a kernel_launcher declaration just before its use"
    decl = "\n\t extern " + kernel_launcher_declarations[m.group(1)] + ";\n" + m.group(0)
    if verbose:
        print "Inserting", decl
    return decl

def patch_opencl_kernel_declaration(matchobj):
    ''' Patch a comma separated list of OpenCL kernel arguments and
    prepend "P4A_accel_global_address" in front of every pointer/array '''
    head = matchobj.group(1)
    # Split declarations string in a list
    decls=re.split(",",matchobj.group(2))
    patched_decls=[]
    # Patch all declarations
    for decl in decls:
        patched_decl=re.sub('\s+((?:\w\s?)+)\s*\*\s*(\w+)', "P4A_accel_global_address \\1 *\\2", decl)
        patched_decl=re.sub('\s+((?:\w\s?)+)\s*(\w+)\s*((?:\[\w*\])+)\s*', "P4A_accel_global_address \\1 \\2 \\3", patched_decl)
        patched_decls+=[patched_decl]
    return head+", ".join(patched_decls)



def replace_by_opencl_own_declarations(content):
    # Replace sinf by the opencl intrinsic sin function
    content = re.sub("sinf","sin",content)
    
    # size_t is not allowed in kernel parameter OpenCL ...
    # size_t can be a 32-bit or a 64-bit unsigned integer, and the OpenCL
    # compiler does not accept variable types that are implementation-dependent
    # for kernel arguments.
    # all typedef.*size_t declaration have been changed using remove_libc_typedef(),
    # Here we remove all of them, even if it's not a kernel argument...
    content = re.sub("size_t","unsigned long int",content)

    # OpenCL pointers and array variable must be explicitely declared
    # in the global memory space
    
    # P4A_accel_kernel.*(.*  type  var[exp]
    # substituted by:
    # P4A_accel_kernel.*(.*  P4A_accel_global_address type  var[exp]
    # P4A_accel_kernel.*(.*  type  *var
    # substituted by:
    # P4A_accel_kernel.*(.*  P4A_accel_global_address type  *var
    content = re.sub("(P4A_accel_kernel(?:_wrapper)?\s*\w*\()([^;\n]*)",patch_opencl_kernel_declaration, content)
    #content = re.sub("(P4A_accel_kernel(?:_wrapper)?\s*\w*\()([^;\n]*)",patch_opencl_kernel_declaration, content)
    content = re.sub("(static [^\n]*?\s*\w*\()([^;\n]*)",patch_opencl_kernel_declaration, content)

    return content


def p4a_launcher_clean_up(match_object):
    """Get a match object during a p4a_launcher and generate a
    P4A_call_accel_kernel_$x$d macro of the p4a_wrapper_... function"""

    if verbose:
        for i in range(9):
            print i,'=', match_object.group(i), '|END|'

    # Get all the interesting pieces from the original PIPS code:
    launcher_definition_header = match_object.group(1)
    before_loop_nest = match_object.group(3)
    iteration_dimension = match_object.group(4)
    iteration_space = match_object.group(5)
    loop_nest = match_object.group(6)
    wrapper_function_name = match_object.group(7)
    wrapper_function_parameters = match_object.group(8)

    # Extract any declaration variables:
    # Variables are anything except plain '{' lines before the loop nest:
    variables_before = re.sub("(?s)\n\\s*{\\s*\n", "\n", before_loop_nest)
    # Remove blank lines:
    variables_before = re.sub("(?s)\n\\s*\n", "\n", variables_before).rstrip()
    if verbose:
        print 'vb', variables_before, 'end'

    # Inside loop nest, just remove the for loops and the attached pragmas if any:
    variables_in_loop_nest = re.sub("(?s)\s*(#pragma[^\n]*\s*)*for\\([^\n]*", "", loop_nest)
    # Remove also the '// To be assigned to a call to P4A_vp_...' that
    # should no be stay here:
    variables_in_loop_nest = re.sub("(?s)\\s*// To be assigned to a call to P4A_vp_[^\n]*", "", variables_in_loop_nest)
    # Remove blank lines:
    variables_in_loop_nest = re.sub("(?s)\n\\s*\n", "\n", variables_in_loop_nest).rstrip()
    if verbose:
        print 'viln', variables_in_loop_nest, 'end'

    # Now construct the final lancher construction:
    launcher = launcher_definition_header + variables_before \
               + variables_in_loop_nest \
               + "\n   P4A_call_accel_kernel_" + iteration_dimension + "d(" \
               + wrapper_function_name +", " \
               + iteration_space + ", " + wrapper_function_parameters +";\n}"
    if verbose:
        print 'launcher =', launcher, '|END|'

    return launcher

def patch_to_use_p4a_methods(file_name, dir_name, includes):
    """This post-process the PIPS generated source files to use the
    Par4All Accel run-time"""

    file_base_name = os.path.basename(file_name);

    if file_base_name == 'p4a_stubs.c':
        if verbose:
            print 'Skip the stubs since we will use the real P4A runtime instead'
        return

    print 'Patching', file_name, 'to directory', dir_name
    # Where we will rewrite the result:
    dest_file_name = os.path.join(dir_name, file_base_name)

    # Read the PIPS output:
    f = open(file_name)
    # slurp all the file in a string:
    content = f.read()
    f.close()

    content = remove_libc_typedef (content)

    if (p4a_util.opencl_file_p(file_base_name)):
        content = replace_by_opencl_own_declarations(content)

    header = ""
    # Inject P4A accel header definitions except in opencl files:
    if (not p4a_util.opencl_file_p(file_base_name)):
        accel_header = """/* Use the Par4All accelerator run time: */
#include <p4a_accel.h>
"""
	header=accel_header
    for include in includes:
        header += '#include "' + include + '"\n'
    
    #OpenCL only; inject P4A_wrapper_proto declaration. Wrapper contains 
    #call to load kernels. Headers and typedef are already in the .c files,
    #generated by pips except p4a_accel.h : 
    #The arguments for P4A_wrapper_xxx in the P4A_wrapper_proto declaration,
    #are currently taken from the .c files
    #(it's not complete) and it's a lazy choice, but it works.
    #If needed it can be taken from .cl file (more complicated)
    opencl_wrappers=re.findall("""(?s)//Opencl wrapper declaration\n(.*?)(?#
    # Get the p4a_wrapper_... call with its arguments:
    )(p4a_wrapper_\\w+)\(([^;]*?)\\;""",
		content)
    if opencl_wrappers:
        wrapper_proto_declaration=""
        for wrapper in opencl_wrappers:
            wrapper_proto_declaration= wrapper_proto_declaration + "P4A_wrapper_proto("+wrapper[1]+", "+wrapper[2]+";\n"
        content= accel_header + wrapper_proto_declaration + content
		#remove p4a_wrapper_xxx and p4a_kernel_xxx declaration generated by pips
		#content = re.sub(r"void\s*p4a_launcher_\w+(.*?)(,\s*p4a_wrapper_\w+\((.*?)\))(.*?)(,\s*p4a_kernel_\w+\((.*?)\))"," \\4", content)
        content = re.sub(r"(\s*P4A_accel_kernel p4a_kernel_\w+\((.*?)\))","", content)
        content = re.sub(r"(\s*P4A_accel_kernel_wrapper p4a_wrapper_\w+\((.*?)\))","", content)
    else:
		content = header + content


    # Clean-up headers and inject standard headers:
    ## content = re.sub("(?s)(/\\*\n \\* file for [^\n]+\n \\*/\n).*/\* Define some macros helping to catch buffer overflows.  \*/",
    ##                  "\\1#include <p4a_accel.h>\n#include <stdio.h>\n#include <math.h>\n", content)

    # Inject run time initialization:
    content = re.sub("// Prepend here P4A_init_accel\n",
                     "P4A_init_accel;\n", content)

    # Now the outliner output all the declarations in one line, so put
    # only one function per line for further replacement:
    #content = re.sub(", (p4a_kernel[^0-9]+[0-9]+\\()",
    #                 ";\nvoid \\1", content)

    # Add accelerator attributes on accelerated parts:
    #content = re.sub("(void p4a_kernel_wrapper_[0-9]+[^\n]+)",
    #                 "P4A_accel_kernel_wrapper \\1", content)
    #content = re.sub("(void p4a_kernel_[0-9]+[^\n]+)",
    #                 "P4A_accel_kernel \\1", content)

    # Generate accelerated kernel calls:
    ## Replace
    ##   // Loop nest P4A begin,2D(500,500)
    ##   for(i = 0; i <= 500; i += 1)
    ##     for(j = 0; j <= 500; j += 1)
    ##       // Loop nest P4A end
    ##       if (i<=500&&j<=500)
    ##         p4a_kernel_wrapper_2(space, i, j);
    ## }
    ## with
    ##   P4A_call_accel_kernel_2D(p4a_kernel_wrapper_2, 500, 500, i, j);
    ## }

    ## There may be some C99 desugared for loop index desugaring to deal
    ## with here too

#   content = re.sub("(?s)// Loop nest P4A begin,(\\d+)D\\(([^)]+)\\).*// Loop nest P4A end\n.*?(p4a_kernel_wrapper_\\d+)\\(([^)]*)\\);\n",
#                     "P4A_call_accel_kernel_\\1d(\\3,\\2,\\4);\n", content)

    ## content = re.sub("""(?s)// Loop nest P4A begin,(\\d+)D\\(([^)]+)\\)(?:.*?)(int \\w+;)?(?(3)(?:.*))(?#
    ##     to skip to the next "Loop nest P4A end", not the last one...
    ##     )// Loop nest P4A end\n.*?(?#
    ##     to skip to "the p4a_wrapper", not the last one...
    ##     )(p4a_wrapper_\\w+)\\(([^;]*)\\;.*?(?#
    ##     no slurp to the next "}" at the begin of a line
    ##     )\n}""",
    ##     p4a_call_accel_kernel_repl, content)
    content = re.sub("""(?s)(void p4a_launcher_(\\w+?)\\([^;]*?(?#
    # Anchor on the declaration because I've not been able to work
    # with a regular expression without it
    )\n{)(.*?)(?#
    # First capture any declaration at the begining up to the loop label:
    )// Loop nest P4A begin,(\\d+?)D\\((.+?)\\)\\n(?#
    # So now we have captured kernel iteration space.
    # Next capture up to following 'Loop nest P4A end', not the last one...
    )(.*?)// Loop nest P4A end.+?(?#
    # Now get the p4a_wrapper_... call with its arguments:
    )(p4a_wrapper_\\w+?)\\(([^;]*?)\\;(?#
    # Then slurp to the next "}" at the begin of a line which is
    # the end of function body:
    ).*?\n}""",
        p4a_launcher_clean_up, content)


    # Get the virtual processor coordinates:
    ## Change
    ##    // To be assigned to a call to P4A_vp_1: j
    ## into
    ##     // Index has been replaced by P4A_vp_1
    ##    j = P4A_vp_1;
    content = re.sub("( *)// To be assigned to a call to (P4A_vp_[0-9]+): ([^\n]+)",
                     "\\1// Index has been replaced by \\2:\n\\1\\3 = \\2;", content)

    # Add missing declarations of the p4a_kernel_launcher (outliner or
    # prettyprinter bug?)
    # It seems to be corrected now
    ### content = re.sub("\n[^\n]+(p4a_kernel_launcher_\\d+)\\(",
    ###                 insert_kernel_launcher_declaration, content)

    # NULL is preprocessed differently in C and C++ ;
    # PIPS generated code for NULL is "(void *) 0"
    # This will break cuda compilation !
    # So here is a quick hack to recover NULL symbolic
    content = re.sub(r'\(void \*\) 0',
                     "NULL", content)

    # flag all static functions as device functions, after splitting them
    # P4A_DEVICE is implemented in p4a_accel headers, depending on the backend
    old_content=""
    while old_content != content:
        old_content=content
        content=re.sub(r"static (\w+) (\w+)(\([^)]+\)), (\w+)(\([^)]+\))",r"static \1 \2\3;\nstatic \1 \4\5", content)
    content = re.sub(r'\nstatic (.*?) p4a_device_(.*?)\n',r'\nstatic P4A_DEVICE \1 p4a_device_\2\n', content)

    if verbose:
        print content,

    if verbose:
        print 'Rewrite the content to', dest_file_name
    f = open(dest_file_name, 'w')
    f.write(content)
    f.close()

    # Save a .cu version too just in case :-)
    dest_file_name = re.sub("\\.c$", ".cu", dest_file_name)
    f = open(dest_file_name, 'w')
    f.write(content)
    f.close()


def main():
    global verbose

    parser = optparse.OptionParser(usage = "usage: %prog [options] <files>",
                                   version = "$Id")
                                   
    parser.add_option("-d", "--dest-dir",
                      action = "store", type = "string",
                      dest = "dest_dir", default = "P4A",
                      help = """The destination directory name to create and
to put files in. It defaults to "P4A" in the current directory>""")

    parser.add_option("--includes", "-I", action = "append", metavar = "header_list", default = [],
                      help = "Specify some includes to be insetred at the begining of the file to be post processed.")
                      
    group = optparse.OptionGroup(parser, "Debug options")

    group.add_option("-v",  "--verbose",
                     action = "store_true", dest = "verbose", default = False,
                     help = "Run in verbose mode")

    group.add_option("-q",  "--quiet",
                     action = "store_false", dest = "verbose",
                     help = "Run in quiet mode [default]")

    parser.add_option_group(group)

    (options, args) = parser.parse_args()

    verbose = options.verbose

    if not os.path.isdir(options.dest_dir):
        # Create the destination directory:
        os.makedirs(options.dest_dir)

    for name in args:
        gather_kernel_launcher_declarations(name)
    if verbose:
        print kernel_launcher_declarations

    for name in args:
        patch_to_use_p4a_methods(name, options.dest_dir, options.includes)

# If this programm is independent it is executed:
if __name__ == "__main__":
    main()
