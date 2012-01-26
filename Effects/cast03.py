
from validation import vworkspace

with vworkspace() as w:
  w.fun.without_cast.display("print_code_regions")
#  w.fun.with_cast.display("print_code_regions")
  
