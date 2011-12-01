from __future__ import with_statement # this is to work with python2.5
from pyps import workspace

w = workspace("properties1.c",deleteOnClose=True)

# mandatory for A&K (Rice) parallization on C files
w.props.memory_effects_only=False

print str (w.props.OMP_MERGE_POLICY)
print str (w.props.PRETTYPRINT_SEQUENTIAL_STYLE)

w.props.PRETTYPRINT_SEQUENTIAL_STYLE = "do"
w.props.OMP_MERGE_POLICY = "inner"

#Here we expect to have some update properites and this should
#work
print str (w.props.OMP_MERGE_POLICY)
print str (w.props.PRETTYPRINT_SEQUENTIAL_STYLE)

#Get foo function
main = w.fun.main

main.privatize_module ()
main.flag_parallel_reduced_loops_with_openmp_directives()
main.internalize_parallel_code ()
main.ompify_code ()

# The merge_pragma work differently depending on the property OMP_MERGE_POLICY
# Here we set the property to "inner" but the value seen from the phase is still
# "outer" which is the default value. As a consequence the pragma are merge on
# the outer loop and not on the inner loop as expected.
main.omp_merge_pragma ()

main.display ()
w.close()
