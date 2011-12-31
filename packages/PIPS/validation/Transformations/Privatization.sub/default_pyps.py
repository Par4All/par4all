from validation import vworkspace

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.props.prettyprint_all_private_variables = True
    w.all_functions.validate_phases("privatize_module");

    c_modules = w.filter(lambda m: not m.compilation_unit_p() and m.language=="c")
    c_modules.validate_phases("localize_declaration")

