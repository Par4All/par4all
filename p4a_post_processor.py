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

    # Add accelerator attributes on accelerated parts:
    content = re.sub("(void p4a_kernel_wrapper_[0-9]+[^\n]+)",
                     "P4A_ACCEL_KERNEL_WRAPPER \\1", content)
    content = re.sub("(void p4a_kernel_[0-9]+[^\n]+)",
                     "P4A_ACCEL_KERNEL \\1", content)

    # Generate accelerated kernel calls:
    ## Replace
    ## // Loop nest P4A begin,2D(500,500)
    ## for(i = 0; i <= 500; i += 1)
    ##   for(j = 0; j <= 500; j += 1)
    ##      // Loop nest P4A end
    ##      if (i<=500&&j<=500)
    ##         p4a_kernel_wrapper_2(space, i, j);
    ##
    ## with
    ## P4A_CALL_ACCEL_KERNEL_2D(p4a_kernel_wrapper_2, 500, 500, i, j);

    ## content = re.sub("// Loop nest P4A begin,2D\(([0-9]+),([0-9]+)\)\n[^\n]+\n[^\n]+\n +// Loop nest P4A end\n[^\n]+\n +(p4a_kernel_wrapper_\\d+)\\(([^)]*)\\)\n",
    ##                  "P4A_CALL_ACCEL_KERNEL_2D(\\3,\\1,\\2,\\4)", content)

    content = re.sub("(?s)// Loop nest P4A begin,2D\((\\d+),(\\d+)\).*// Loop nest P4A end\n[^\n]+\n +(p4a_kernel_wrapper_\\d+)\\(([^)]*)\\);\n",
                     "P4A_CALL_ACCEL_KERNEL_2D(\\3,\\1,\\2,\\4);\n", content)

    # Compatibility
    content = re.sub("P4A_VP_0\\(\\)", "P4A_VP_X", content)
    content = re.sub("P4A_VP_1\\(\\)", "P4A_VP_Y", content)

    # Clean-up headers and inject standard header injection:
    content = re.sub("(?s)(/\\*\n \\* file for [^\n]+\n \\*/\n).*extern int getloadavg\\(double __loadavg\\[], int __nelem\\);",
                     "\\1#include <stdio.h>\n", content)

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
