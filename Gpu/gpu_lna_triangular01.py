from validation import vworkspace


with vworkspace() as w:
  w.props.memory_effects_only = False
  w.fun.main.internalize_parallel_code()
  w.fun.main.gpu_loop_nest_annotate()
  w.all_functions.display()

