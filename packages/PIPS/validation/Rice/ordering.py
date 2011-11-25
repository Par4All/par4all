from validation import vworkspace

with vworkspace() as w:
  w.activate("MUST_REGIONS")
  w.props.memory_effects_only=False
  w.fun.pb_ReadParameters.internalize_parallel_code()
  w.fun.pb_ReadParameters.validate_phases("rice_semantics_dependence_graph")

