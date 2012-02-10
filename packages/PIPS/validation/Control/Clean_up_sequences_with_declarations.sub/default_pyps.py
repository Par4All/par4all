from validation import vworkspace
import os

#os.environ["CLEAN_UP_SEQUENCES_DEBUG_LEVEL"]="8"
#os.environ["CONTROL_DEBUG_LEVEL"]="8"
with vworkspace() as w:
    w.props["CLEAN_UP_SEQUENCES_WITH_DECLARATIONS"]=True
    w.all_functions.display()
    #w.all_functions.validate_phases("merge_sequences");

