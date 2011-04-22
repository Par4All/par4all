import workspace_git
import pyps

with workspace_git.git("prog.c", deleteOnClose=True) as w:
	w.fun.main.partial_eval()
