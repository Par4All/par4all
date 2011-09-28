from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.fun.main.validate_phases("simplify_control_directly")

