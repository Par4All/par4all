#! /usr/bin/env python

"""
Convert the PIPS C output to a P4A output

Ronan.Keryell@hpc-project.com
"""

import sys, re

def patch_to_use_p4a_methods(file_name):
    print 'Patching', file_name
    # Parse the PIPS output:
    f = open(file_name)
    # slurp all the file in a string:
    content = f.read()
    f.close()

    ## Change
    ##    // To be replaced with a call to P4A_VP_1
    ##    j = j;
    ## into
    ##     // Index has been replaced by P4A_VP_1
    ##    j = P4A_VP_1();
    content = re.sub("( *)// To be replaced with a call to (P4A_VP_[0-9]+)\n[^=]+= ([^;]+)",
                     "\\1// Index has been replaced by \\2\n\\1\\3 = \\2()", content)

    # Insert a
    #include <p4a_accel.h>
    content = re.sub("^",
                     "#include <p4a_accel.h>\n", content)

    # Compatibility
    content = re.sub("// Prepend here P4A_INIT_ACCEL\n",
                     "P4A_INIT_ACCEL;\n", content)

    # Compatibility
    content = re.sub("P4A_VP_0", "P4A_VP_X", content)
    content = re.sub("P4A_VP_1", "P4A_VP_Y", content)

    print content,

    # Rewrite the content:
    f = open(file_name, 'w')
    f.write(content)
    f.close()
    # Save a .cu version too just in case :-)
    file_name = re.sub("\\.c$", ".cu", file_name)
    f = open(file_name, 'w')
    f.write(content)
    f.close()

for name in sys.argv[1:]:
    patch_to_use_p4a_methods(name)
