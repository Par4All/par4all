from validation import vworkspace

with vworkspace() as w:
    w.props.memory_effects_only = False
    w.fun.main.validate_phases("privatize_module", "internalize_parallel_code", "localize_declaration");

