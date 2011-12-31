from validation import vworkspace as workspace
w=workspace()
w.props.prettyprint_sequential_style = "do"
w.fun.main.privatize_module()
w.fun.main.flag_parallel_reduced_loops_with_openmp_directives()
w.fun.main.display()
w.close()
