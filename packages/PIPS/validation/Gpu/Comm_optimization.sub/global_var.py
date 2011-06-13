from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.fun.kernel.flag_kernel()
    w.fun.main.kernel_data_mapping()
    w.all_functions.display()

