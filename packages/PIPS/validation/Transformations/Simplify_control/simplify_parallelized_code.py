from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.props.semantics_compute_transformers_in_context = False
    w.all_functions.internalize_parallel_code()
    w.all_functions.validate_phases("simplify_control")
    w.props.semantics_compute_transformers_in_context = True
    w.all_functions.internalize_parallel_code()
    w.all_functions.validate_phases("simplify_control")

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.props.semantics_compute_transformers_in_context = True
    w.all_functions.internalize_parallel_code()
    w.all_functions.validate_phases("simplify_control")

