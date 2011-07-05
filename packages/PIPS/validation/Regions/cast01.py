from validation import vworkspace
import openmp

with vworkspace() as w:
    w.all_functions.display('print_code_proper_effects')
    w.all_functions.display('print_code_regions');
    w.all_functions.validate_phases('openmp')
