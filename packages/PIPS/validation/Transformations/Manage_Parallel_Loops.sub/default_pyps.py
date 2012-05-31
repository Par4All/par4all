from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace


#import os
#os.environ["SCALARIZATION_DEBUG_LEVEL"]="5"
with vworkspace() as w:
    w.props.memory_effects_only = False
    w.props.prettyprint_all_private_variables = True
    w.all_functions.validate_phases("privatize_module")
    w.props.prettyprint_all_private_variables = False
    w.all_functions.validate_phases("internalize_parallel_code")
    w.all_functions.validate_phases("limit_parallelism_using_complexity")
