from validation import vworkspace


with vworkspace() as w:
  w.props.memory_effects_only = False
  w.all_functions.internalize_parallel_code()
  w.all_functions.gpu_loop_nest_annotate()
  w.all_functions.display()

