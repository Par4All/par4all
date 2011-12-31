#!/usr/bin/env python

import sys, os, shutil

from pyps import *

# removing previous output
if os.path.isdir("loop_tiling.database"):
    shutil.rmtree("loop_tiling.database", True)

ws = workspace("loop_tiling.c", name="loop_tiling",deleteOnClose=True)

ws.props.ABORT_ON_USER_ERROR = True

fct = ws.fun.main
# try a basic function
fct.privatize_module ()

# do some tiling on loop
print "do some tiling on loop"
#look for the desired loop
for loop in fct.loops ():
    lbl = loop.label
    if lbl == "l300":
        loop.loop_tiling (LOOP_TILING_MATRIX = "1111 0 0 , 0 2222 0 , 0 0 3333")

# print result
fct.display ()

# close the workspace
ws.close()
