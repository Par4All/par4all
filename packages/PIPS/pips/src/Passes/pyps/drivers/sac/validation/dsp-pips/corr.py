import pyps
import sac

pyps.workspace.delete("corr")
with pyps.workspace("corr.c", "tools.c", parents=[sac.workspace],
		deleteOnClose=False,name="corr") as w:
	c=w.fun.corr
	c.sac(verbose=True,load_zeros=True)
	c.display()
	w.goingToRunWith(w.save(rep="corr-sac"),"corr-sac")
