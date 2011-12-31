from __future__ import with_statement
from validation import vworkspace
with vworkspace() as w:
    w.props.linearize_array_use_pointers=True
    w.props.linearize_array_skip_local_arrays=True
    w.fun.pain.validate_phases("linearize_array")
