from __future__ import with_statement
from validation import vworkspace

with vworkspace() as w:
    w.props.constant_path_effects=False
    w.all_functions.display(activate="print_code_proper_effects")
