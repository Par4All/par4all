
from validation import vworkspace

with vworkspace() as w:
  w.fun.main.display("print_code_proper_effects")
  w.fun.main.display("print_code_regions")
#  w.fun.with_cast.display("print_code_regions")
  
