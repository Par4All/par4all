from validation import vworkspace
import pypsex
#import os

with vworkspace() as w:
    #os.environ['CHAINS_DEBUG_LEVEL']='5'
    w.props.prettyprint_statement_number=True
    w.props.must_regions = True
    w.props.constant_path_effects = False
    w.props.parallelization_ignore_thread_safe_variables = True
    w.props.prettyprint_all_private_variables = True
    w.fun.main.validate_phases("privatize_module");

    w.props.prettyprint_all_private_variables = False;
    w.fun.main.validate_phases("coarse_grain_parallelization");

