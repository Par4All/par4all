from validation import vworkspace

with vworkspace() as w:
    w.props.linearize_array_use_pointers=True
    w.all_functions.display()
    w.all_functions.validate_phases("linearize_array")

