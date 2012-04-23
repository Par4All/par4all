from validation import vworkspace

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.props.prettyprint_all_private_variables = True
    w.all_functions.validate_phases("privatize_module_even_globals");


