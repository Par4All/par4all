import workspace_git
import pyps

with workspace_git.workspace("prog.c", deleteOnClose=False) as w:
	w.fun.main.partial_eval()
	w.fun.main.flatten_code()
	w.fun.main.partial_eval()
	w.fun.main.suppress_dead_code()
