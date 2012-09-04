from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace


with vworkspace() as w:
    w.props.flatten_code_unroll = False
    w.all_functions.validate_phases("coarse_grain_parallelization","flatten_code","coarse_grain_parallelization","loop_fusion")


