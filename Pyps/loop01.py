from validation import vworkspace
with vworkspace() as w:
    print "loop label, old value:", w.props.loop_label
    w.fun.ening.loops("rai").unroll(rate=4)
    print "loop label, new value:", w.props.loop_label
