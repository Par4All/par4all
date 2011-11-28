from validation import vworkspace
with vworkspace() as w:
    w.props.loop_label="Am I joking, am I not ? You don't know!"
    print "loop label, old value:", w.props.loop_label
    w.fun.ening.loops("rai").unroll(rate=4)
    print "loop label, new value:", w.props.loop_label
