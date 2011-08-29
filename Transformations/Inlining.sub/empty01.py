from validation import vworkspace
import pypsex

with vworkspace(deleteOnClose=False) as w:
  w.fun.empty.inlining()
  w.fun.caller.display()
