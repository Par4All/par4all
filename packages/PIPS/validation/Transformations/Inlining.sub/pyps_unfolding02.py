from validation import vworkspace
import pypsex

with vworkspace() as w:
  w.fun.unfold.validate_phases('unfold')
  w.fun.non_unfold.display()


