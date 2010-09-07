#!/usr/bin/env python

import sys, os, shutil

from pyps import *

# removing previous output
if os.path.isdir("effects.database"):
    shutil.rmtree("effects.database", True)

ws = workspace(["effects.c"], name="effects")

ws.set_property(ABORT_ON_USER_ERROR = True)

print "cumulated effects on only one function"
fct = ws.fun.add_comp
fct.print_code_cumulated_effects()
fct.display ()

# close the workspace
ws.close()
