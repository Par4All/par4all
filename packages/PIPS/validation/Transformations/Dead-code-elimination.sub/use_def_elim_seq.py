from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.fun.USE_DEF_ELIM_SEQ.validate_phases("dead_code_elimination")

