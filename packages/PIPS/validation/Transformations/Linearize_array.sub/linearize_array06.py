from validation import vworkspace
import ir_navigator

with vworkspace() as w:
  w.all_functions.linearize_array(use_pointers=True,vla_only=True)
  w.all_functions.display()

