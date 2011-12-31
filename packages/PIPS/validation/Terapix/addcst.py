from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
from terapyps import workspace as teraw, Maker
workspace.delete("addcst")
with teraw("addcst.c", name="addcst", deleteOnClose=False, recoverInclude=False) as w:
    for f in w.fun:
        f.terapix_code_generation(debug=True)
#    w.compile(Maker())
