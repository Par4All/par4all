from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.activate('must_regions')
    w.props.memory_effects_only = False
    
    # Force the loop nest to be parallel :-)
    w.fun.main.loops()[0].parallel=True
    w.fun.main.loops()[0].loops()[0].parallel=True
    
    w.props.SCALARIZATION_KEEP_PERFECT_PARALLEL_LOOP_NESTS=True
    w.all_functions.validate_phases("scalarization")

    w.props.SCALARIZATION_KEEP_PERFECT_PARALLEL_LOOP_NESTS=False
    w.all_functions.validate_phases("scalarization")

