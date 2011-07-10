#! /usr/bin/env python

"""
Convert the PIPS C output to a P4A output

Ronan.Keryell@hpc-project.com
"""

#import string, re, sys, os, types, optparse

import sys, re, os, optparse, subprocess

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

    # Inside loop nest, just remove the for loops:
    variables_in_loop_nest = re.sub("(?s)\\s*for\\([^\n]*", "", loop_nest)
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

    # Inject P4A accel header definitions:
    header = """/* Use the Par4All accelerator run time: */
#include <p4a_accel.h>
"""
    for include in includes:
        header += '#include "' + include + '"\n'
    content = header + content

    # Clean-up headers and inject standard headers:
    ## content = re.sub("(?s)(/\\*\n \\* file for [^\n]+\n \\*/\n).*/\* Define some macros helping to catch buffer overflows.  \*/",
    ##                  "\\1#include <p4a_accel.h>\n#include <stdio.h>\n#include <math.h>\n", content)

    # Inject run time initialization:
    content = re.sub("// Prepend here P4A_init_accel\n",
                     "P4A_init_accel;\n", content)

	# This patch is a temporary solution. It may not cover all possible cases
    # I guess all this should be done by applying partial_eval in PIPS ?
	# It replaces some array declarations that nvcc do not compile.
	# For ex.
	# int n = 100; double a [n];
	# is replaced by :
	# double a[100]
    fObj = re.findall("\s*(?:int|\,)\s*(\w+)\s*=\s*(\d+)", content)
    for obj in fObj:
		content = re.sub("\["+obj[0]+"\]","["+obj[1]+"]", content)

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

    content = remove_libc_typedef (content)

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
