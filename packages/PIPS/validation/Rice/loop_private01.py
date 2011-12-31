from validation import vworkspace
import openmp

with vworkspace() as w:
  w.props.memory_effects_only = False
  w.props.constant_path_effects = False
    
  w.fun.loop_private.validate_phases("internalize_parallel_code")

