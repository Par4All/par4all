from validation import vworkspace
with vworkspace() as w:
    w.props.outline_module_name="ninja_turtle"
    w.props.outline_label="not_me"
    w.props.outline_allow_globals=True
    w.all_functions.validate_phases("outline")
    w.fun.ninja_turtle.display()
