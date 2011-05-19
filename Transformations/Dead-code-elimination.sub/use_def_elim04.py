from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace


with vworkspace() as w:
    w.props.memory_effects_only = False
    w.props.prettyprint_memory_effects_only = True
    w.props.prettyprint_scalar_regions = True

    w.activate("must_regions")
    w.fun.use_def_elim04.validate_phases("print_code_regions",
                                         "print_code_in_regions",
                                         "print_code_out_regions",
                                         "dead_code_elimination")


