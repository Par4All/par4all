import pyps
from sys import argv
if len(argv) == 1:
    w=pyps.workspace("broker01.c")
    w.props.preprocessor_missing_file_handling="query"
    w.props.preprocessor_missing_file_generator="python broker01.py"
    w.fun.main.display(pyps.module.print_code_cumulated_effects)
else:
    # tricky, use the test file as a simple broker too :p
    print "foo.c"
