from validation import vworkspace

with vworkspace() as w:
  w.props.constant_path_effects = False
  w.fun.init.display("print_code_cumulated_effects")
  w.fun.cast_implicit.display("print_code_proper_effects")
