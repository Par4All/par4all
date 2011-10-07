from __future__ import with_statement # this is to work with python2.5
import re
from validation import vworkspace

with  vworkspace() as w:

    w.fun.main.coarse_grain_parallelization()    
    w.fun.main.gpu_ify(gpu_use_wrapper = False,
                            gpu_use_KERNEL = False,
                            gpu_use_launcher = True,
                            gpu_use_launcher_independent_compilation_unit = True)


    modules = w.filter(lambda m: m.name.startswith("p4a_launcher"))
    modules.coarse_grain_parallelization()    

    # this used to fail during the "save()" because the new compilation_unit_name wasn't well computed !
    modules.gpu_ify(gpu_use_wrapper = True,
                            gpu_use_KERNEL = True,
                            gpu_use_launcher = False,
                            gpu_use_kernel_independent_compilation_unit = True)

    ( saved, headers ) = w.save()

    for fname in saved:
        print "//"
        print "// Display saved file " + fname
        print "//"
        with open(fname, 'r') as f:
            print f.read()


