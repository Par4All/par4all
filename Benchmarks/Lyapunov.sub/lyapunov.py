from __future__ import with_statement
from pyps import workspace, module
from glob import glob
workspace.delete("lyapunov")
with workspace(*glob("*.c"), name='lyapunov') as w:
    w.props.PREPROCESSOR_MISSING_FILE_HANDLING = "generate"
    w.all_functions.display()
    w.all_functions.print_code_proper_effects()
    w.all_functions.print_code_cumulated_effects()
    w.all_functions.print_code_transformers()
    w.all_functions.print_code_preconditions()
    w.activate(module.must_regions)
    w.all_functions.print_code_regions()
