from validation import vworkspace

# Without semantics_compute_transformers_in_context
with vworkspace() as w:
  w.activate("MUST_REGIONS")
  w.props.constant_path_effects=False
  w.props.memory_effects_only=False
  w.fun.main.display("print_code_transformers")

# With semantics_compute_transformers_in_context
with vworkspace() as w:
  w.activate("MUST_REGIONS")
  w.props.constant_path_effects=False
  w.props.memory_effects_only=False

  w.props.semantics_compute_transformers_in_context=True
  w.fun.main.display("print_code_transformers")


