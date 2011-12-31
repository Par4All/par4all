from validation import vworkspace


with vworkspace("include/p4a_stubs.c") as w:
   w.props.memory_effects_only = False
   w.props.constant_path_effects = False
   w.props.isolate_statement_even_non_local=True
   w.activate("must_regions")
   m=w.fun.main
   m.privatize_module()
   m.internalize_parallel_code()
   m.coarse_grain_parallelization()
   m.gpu_ify() 
   w.fun.p4a_launcher_main.kernel_load_store()   
   m.display()

