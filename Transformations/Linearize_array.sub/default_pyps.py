from validation import vworkspace

with vworkspace() as w:
  w.all_functions.linearize_array(LINEARIZE_ARRAY_USE_POINTERS=True,LINEARIZE_ARRAY_VLA_ONLY=True)
  w.all_functions.display()

