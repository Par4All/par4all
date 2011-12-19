from validation import vworkspace
with vworkspace() as w:
    w.all_functions.display()
    w.all_functions.validate_phases("split_initializations")
