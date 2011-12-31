from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.fun.Run.validate_phases("print_code")
    w.fun.Run.validate_phases("simplify_control")
    w.fun.Run.validate_phases("simplify_control")

