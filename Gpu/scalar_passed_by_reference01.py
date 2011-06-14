from validation import vworkspace


with vworkspace() as w: 
  w.props.memory_effects_only = False
  w.fun.main.scalarization()
  w.fun.main.privatize_module()
  w.fun.main.internalize_parallel_code()
  w.fun.main.coarse_grain_parallelization()
  w.fun.main.internalize_parallel_code()
  w.fun.main.coarse_grain_parallelization()
  w.fun.main.gpu_ify()
  w.all_functions.display()

