from validation import vworkspace
#import os

with vworkspace() as w:
    #os.environ['PROPER_EFFECTS_DEBUG_LEVEL'] = '8'
    w.all_functions.display(activate="print_code_proper_effects")
