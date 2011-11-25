from validation import vworkspace

with vworkspace() as w:
  w.all_functions.linearize_array(use_pointers=True,skip_static_length_arrays=True)
  w.all.display()
  w.all_functions.linearize_array(use_pointers=True,skip_static_length_arrays=False)
  w.all.display()

