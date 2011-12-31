from validation import vworkspace
with vworkspace() as w:
    w.all_functions.validate_phases("print_code")
