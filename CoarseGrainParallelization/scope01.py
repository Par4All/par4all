


import pyps


w= pyps.workspace('scope01.c')

r = w.fun.Run

r.privatize_module()
r.coarse_grain_parallelization()
r.display()


