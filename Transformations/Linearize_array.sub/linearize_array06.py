from validation import vworkspace
import ir_navigator

with vworkspace() as w:
  w.all_functions.linearize_array(use_pointers=True,vla_only=True)
  w.all.display()
  w.all_functions.linearize_array(use_pointers=True,vla_only=False)
  w.all.display()

