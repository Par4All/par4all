from validation import vworkspace

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.all_functions.loop_fusion_with_regions(loop_fusion_max_fused_per_loop=4)
    w.all_functions.display()

