from validation import vworkspace
import validate_fusion

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.all_functions.validate_fusion(parallelize=True)
    w.props.loop_fusion_keep_perfect_parallel_loop_nests = False
    w.all_functions.validate_fusion(parallelize=True)

