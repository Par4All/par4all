from validation import vworkspace
import pypsex

with vworkspace() as w : 
    w.fun.MAD.display()
    w.fun.MAD.unfolding()
    w.fun.MAD.display()
