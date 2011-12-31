from validation import vworkspace

with vworkspace() as w:
  w.props.constant_path_effects = False
  w.props.memory_effects_only = False
  w.fun.main.display("print_code_proper_effects")

