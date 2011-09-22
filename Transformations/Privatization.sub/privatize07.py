from validation import vworkspace
from pyps import module
with vworkspace() as w:
   w.props.memory_effects_only = False
   w.props.constant_path_effects = False
   w.props.prettyprint_all_private_variables=True
   m=w.fun.main
   m.privatize_module()
   m.display(module.print_code_cumulated_effects)
