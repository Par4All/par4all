from validation import vworkspace
import pypsex

with vworkspace() as w:
  w.fun.empty.inlining()
  w.fun.caller.display()
