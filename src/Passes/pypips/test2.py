from pyps import *
ws=workspace(["test2.c"])
conv=ws["convol"]
conv.unfolding()
lbl=conv.loops()[1].label
conv.outline(label=lbl,module_name=(conv.name+"_outlined"))
ws.all.display()
