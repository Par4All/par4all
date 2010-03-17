#!/usr/bin/env python
from pyps import *
ws=workspace(["basics2.c"])
conv=ws["convol"]
conv.unfolding()
lbl=conv.loops()[1].label
conv.outline(label=lbl,module_name=(conv.name+"_outlined"))
ws.all.display()
ws.close()
