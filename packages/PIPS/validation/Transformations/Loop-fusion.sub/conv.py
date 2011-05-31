from validation import vworkspace

with vworkspace() as w:
    c=w.fun.conv
    w.props.loop_fusion_greedy = True
    c.loops(0).unroll(rate=4)
    c.validate_phases("partial_eval","flatten_code","loop_fusion")

