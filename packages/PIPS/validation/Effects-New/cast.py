from validation import vworkspace

with vworkspace() as w:
  w.props.constant_path_effects = False
  w.fun.main.display("print_code_proper_effects")
  w.fun.main.display("print_code_cumulated_effects")
