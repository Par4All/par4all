from validation import vworkspace
#import os
with vworkspace() as w:
#    os.environ['PROPER_EFFECTS_DEBUG_LEVEL']='8'
    w.props.memory_effects_only = False
    w.fun.main.validate_phases("print_code_proper_effects")

