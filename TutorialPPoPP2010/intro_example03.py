from pyps import *

# create the pips workspace
w = workspace(["intro_example03.f"])

w.set_property( PRETTYPRINT_STATEMENT_NUMBER=False)
w.activate('MUST_REGIONS')

w.all.array_bound_check_top_down()

# Propagate constants
w.all.partial_eval()

w.all.display(With='PRINTED_FILE')

# It is always nice to get the new version of the C file
w.all.unsplit()


