#!/usr/bin/env python

import sys, os, shutil

from pyps import *

# removing previous output
if os.path.isdir("loop_tiling.database"):
    shutil.rmtree("loop_tiling.database", True)

ws = workspace(["loop_tiling.c"], name="loop_tiling")

ws.set_property(ABORT_ON_USER_ERROR = True)
#prepare property for tiling
ws.set_property (LOOP_LABEL = "l300")
ws.set_property (LOOP_TILING_MATRIX = "1111 0 0 , 0 2222 0 , 0 0 3333")

fct = ws.fun.main
# try a basic function
fct.privatize_module ()

# do some tiling on loop
print "do some tiling on loop"
fct.loop_tiling ()

# print result
fct.display ()

# close the workspace
ws.close()
