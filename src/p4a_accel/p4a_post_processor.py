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
        assert (len (match_l) == 1)
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

#This function is used by re.sub(...) inside the function patch_to_use_p4a_methods() 
#and builds the accel kernel call statement for
#p4a_launcher_xxx() function (xxx is the name of the module(function))
def p4a_call_accel_kernel_repl(match_object):
	if match_object.group(3):
		return match_object.group(3)+"\n P4A_call_accel_kernel_"+match_object.group(1)+\
			"d("+match_object.group(4)+", "+match_object.group(2)+", "+match_object.group(5)+";\n}"
	else:
		return "P4A_call_accel_kernel_"+match_object.group(1)+"d("+match_object.group(4)+\
			", "+match_object.group(2)+", "+match_object.group(5)+";\n}"

def patch_to_use_p4a_methods(file_name, dir_name, includes):
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
#   content = re.sub("(?s)// Loop nest P4A begin,(\\d+)D\\(([^)]+)\\).*// Loop nest P4A end\n.*?(p4a_kernel_wrapper_\\d+)\\(([^)]*)\\);\n",
#                     "P4A_call_accel_kernel_\\1d(\\3,\\2,\\4);\n", content)

    content = re.sub("""(?s)// Loop nest P4A begin,(\\d+)D\\(([^)]+)\\)(?:.*?)(int \\w+;)?(?(3)(?:.*))(?#
        to skip to the next "Loop nest P4A end", not the last one...
        )// Loop nest P4A end\n.*?(?#
        to skip to "the p4a_wrapper", not the last one...
        )(p4a_wrapper_\\w+)\\(([^;]*)\\;.*?(?#
        no slurp to the next "}" at the begin of a line
        )\n}""",
        p4a_call_accel_kernel_repl, content)

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
