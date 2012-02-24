from validation import vworkspace as workspace
with workspace() as w:
    w.props.CONSTANT_PATH_EFFECTS=False
    w.props.SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT=True
    w.props.UNSPAGHETTIFY_TEST_RESTRUCTURING=True
    w.props.SEMANTICS_FIX_POINT_OPERATOR="derivative"
    w.props.UNSPAGHETTIFY_RECURSIVE_DECOMPOSITION=True
    w.props.ALIASING_ACROSS_IO_STREAMS=False
    w.props.MEMORY_EFFECTS_ONLY=False
    w.fun.lyapunov_finish.display(activate="print_code_proper_effects")
