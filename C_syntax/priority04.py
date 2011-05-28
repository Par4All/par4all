from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w.fun.main.display()
    # The following line is just for design purpose, should be removed before validating the test case
    print "Result of the execution : " + w.compile_and_run()

