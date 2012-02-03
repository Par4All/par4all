from validation import vworkspace
#import os

#os.environ["MERGE_SEQUENCES_DEBUG_LEVEL"]="5"

with vworkspace() as w:
    w.all_functions.display()
    w.all_functions.validate_phases("merge_sequences");

