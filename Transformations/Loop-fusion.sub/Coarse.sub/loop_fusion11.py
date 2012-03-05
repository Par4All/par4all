from validation import vworkspace

with vworkspace() as w:
    w.props.memory_effects_only = True
    w.props.loop_fusion_keep_perfect_parallel_loop_nests = True
    w.all_functions.flatten_code(unroll=False)
    w.all_functions.privatize_module()
    w.all_functions.coarse_grain_parallelization()
    w.all_functions.validate_phases("coarse_grain_parallelization","loop_fusion_with_regions")

