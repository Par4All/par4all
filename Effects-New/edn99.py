from validation import vworkspace


with vworkspace("edn99_ppm.c",cppflags="-I.") as w:
  w.props.constant_path_effects = False
  w.all_functions.display()
  w.all_functions.privatize_module()

