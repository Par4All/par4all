from validation import vworkspace


with vworkspace("include/p4a_stubs.c") as w:
  w.activate("MUST_REGIONS")
  w.activate("TRANSFORMERS_INTER_FULL")
  w.activate("INTERPROCEDURAL_SUMMARY_PRECONDITION")
  w.activate("PRECONDITIONS_INTER_FULL")
  w.props.semantics_trust_array_references = True

  w.fun.main.display("print_code_regions")
  w.fun.kernel.kernel_load_store()

  w.fun.main.display()
  w.fun.kernel.display()

