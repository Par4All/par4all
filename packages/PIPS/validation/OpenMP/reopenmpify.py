from validation import vworkspace

with vworkspace() as w:
    w.props.prettyprint_sequential_style = "do"
    w.fun.main.coarse_grain_parallelization()
    for i in range(10):
        w.fun.main.ompify_code()
    print "# after 10 calls to ompify_code"
    w.fun.main.display()
