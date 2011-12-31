from __future__ import with_statement # this is to work with python2.5
from pyps import workspace
from sac import workspace as sworkspace
from os import remove
from sys import argv
from os.path import basename, splitext
(filename,_)=splitext(basename(argv[0]))
workspace.delete(filename)
with sworkspace(filename+".c", driver="sse", deleteOnClose=True,name=filename) as w:
    m=w[filename]
    m.display()
    m.sac()
    m.display()
    a_out0=w.compile()
    w.run(a_out0)
    a_out1=w.compile(w.get_sac_maker()())
    w.run(a_out1)

