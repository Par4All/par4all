import workspace_git
import pyps

with workspace_git.workspace("prog.c", deleteOnClose=False) as w:
	w.fun.main.partial_eval()
