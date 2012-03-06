from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace
import validate_fusion
import pypsex

with vworkspace() as w:
    w.all_functions.validate_phases("loop_fusion_with_regions")
    w.props.loop_fusion_greedy=True
    print "// w.props.loop_fusion_greedy=True"
    w.all_functions.validate_phases("loop_fusion_with_regions")

