from __future__ import with_statement # this is to work with python2.5
from pyrops import pworkspace



w2 = pworkspace("basics0.c",deleteOnClose=True)
w1 = pworkspace("cat.c",deleteOnClose=True)


# Should display only functions from w1
w1.all_functions.display()

w1.close()
w2.close()
