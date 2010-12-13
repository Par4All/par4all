from __future__ import with_statement # this is to work with python2.5
from pyps import workspace

w = workspace("properties1.c",deleteOnClose=True)


print str (w.props.OMP_MERGE_POLICY)
print str (w.props.PRETTYPRINT_SEQUENTIAL_STYLE)

w.props.PRETTYPRINT_SEQUENTIAL_STYLE = "do"
w.props.OMP_MERGE_POLICY = "inner"

print str (w.props.OMP_MERGE_POLICY)
print str (w.props.PRETTYPRINT_SEQUENTIAL_STYLE)

#Get foo function
main = w.fun.main

main.privatize_module ()
main.internalize_parallel_code ()
main.ompify_code ()
main.omp_merge_pragma ()

main.display ()
