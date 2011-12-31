from validation import vworkspace

with vworkspace() as w:
    w.all_functions.validate_phases("simplify_constant_address_expressions")
