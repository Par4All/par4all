from validation import vworkspace
import pypsex

with vworkspace() as w:
  w.fun.flgr1d_arith_add_fgINT32.validate_phases('unfold')
