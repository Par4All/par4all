import pyps
from sys import argv
if len(argv) == 1:
    # this source include a main() with a call to foo() ; but foo() isn't defined !
    w=pyps.workspace("broker01.c")
    
    # We give a method to resolve missing module (here foo())
    w.props.preprocessor_missing_file_handling="external_resolver"
    w.props.preprocessor_missing_file_generator="python broker01.py"
    
    # We display with cumulated effects because we want callees to be computed
    w.fun.main.display(pyps.module.print_code_cumulated_effects)
else:
    # tricky, use the test file as a simple broker too :p
    print "foo.c"
