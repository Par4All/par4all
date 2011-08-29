from validation import vworkspace


with vworkspace() as w:
  w.props.memory_effects_only = False
  w.props.gpu_loop_nest_annotate_parallel = False
  w.all_functions.validate_phases("gpu_loop_nest_annotate")

