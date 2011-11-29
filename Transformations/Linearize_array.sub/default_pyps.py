from validation import vworkspace

with vworkspace() as w:
  w.all_functions.linearize_array(use_pointers=True)
  w.all_functions.display()

