from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace


with vworkspace() as w:
    w.props.memory_effects_only = False

    w.all_functions.validate_phases("dead_code_elimination","unspaghettify","suppress_dead_code")


