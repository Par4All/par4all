from __future__ import with_statement
from pyps import workspace, module
with workspace("hyantes.c", "options.c") as w:
    w.props.constant_path_effects=False
    w.all_functions.display(module.print_code_proper_effects)
