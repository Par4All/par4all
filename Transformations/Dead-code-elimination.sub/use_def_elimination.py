from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace


with vworkspace() as w:
    w.props.PRETTYPRINT_FINAL_RETURN = True
    w.props.PRETTYPRINT_ALL_LABELS  = True
    w.props.PRETTYPRINT_EMPTY_BLOCKS  = True
    w.props.PRETTYPRINT_UNSTRUCTURED  = True
    w.props.memory_effects_only = False

    w.all_functions.validate_phases("unspaghettify","suppress_dead_code","dead_code_elimination",display_initial=True)


