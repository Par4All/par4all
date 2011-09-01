from validation import vworkspace
#from os import environ

with vworkspace() as w:
#    environ['PROPER_EFFECTS_DEBUG_LEVEL'] = '8'
    w.props.constant_path_effects=False
    w.props.memory_effects_only=True
    w.all_functions.display(activate="print_code_proper_effects")
