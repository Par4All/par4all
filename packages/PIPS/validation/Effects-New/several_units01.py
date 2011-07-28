from validation import vworkspace
from os import environ

with vworkspace("several_units01/foo.c",cppflags="-Iseveral_units01/") as w:
    w.props.constant_path_effects=True
    w.props.aliasing_across_io_streams= False
    w.props.constant_path_effects= False
    w.props.prettyprint_sequential_style= "omp"
    w.props.memory_effects_only = False
    w.activate("must_regions")

    w.all_functions.validate_phases("print_code_proper_effects")
    w.all_functions.validate_phases("print_code_cumulated_effects")

