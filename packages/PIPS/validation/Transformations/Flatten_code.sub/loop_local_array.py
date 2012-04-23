from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.props.constant_path_effects = False

    m = w.fun.main
    m.privatize_module()

    m.coarse_grain_parallelization()
    m.display()
    w.props.flatten_code_unroll=False
    w.all_functions.validate_phases("flatten_code")

