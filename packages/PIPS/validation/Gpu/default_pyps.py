from validation import vworkspace


with vworkspace("include/p4a_stubs.c") as w:
  w.props.linearize_array_use_pointers = True
  w.props.linearize_array_cast_at_call_site = True
  w.props.isolate_statement_even_non_local = True

  # Transform for GPU
  w.all_functions.privatize_module()
  w.all_functions.coarse_grain_parallelization()
  w.all_functions.gpu_ify()

  # Display kernels
  w.filter(lambda m: m.name.startswith("p4a_kernel")).display()

  # Display wrappers
  w.filter(lambda m: m.name.startswith("p4a_wrapper")).display()
  
  # Display launchers
  l=w.filter(lambda m: m.name.startswith("p4a_launcher"))
  l.display()

  # Isolate statement
  l.kernel_load_store()
  l.callers.display()

