from validation import vworkspace


with vworkspace("include/p4a_stubs.c") as w:
  w.activate("MUST_REGIONS")
  w.activate("TRANSFORMERS_INTER_FULL")
  w.activate("INTERPROCEDURAL_SUMMARY_PRECONDITION")
  w.activate("PRECONDITIONS_INTER_FULL")

  w.all_functions.privatize_module(concurrent=True)
  w.all_functions.coarse_grain_parallelization(concurrent=True)
  w.all_functions.gpu_ify();
  w.filter(lambda m: m.name.startswith("p4a_launcher")).kernel_load_store()

  w.all_functions.display()

