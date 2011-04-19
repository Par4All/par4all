


import pyps



w= pyps.workspace('scope01.c')


r = w.fun.Run


print "******** Without points-to analysis ********"

r.privatize_module()
r.coarse_grain_parallelization()
r.display()

print "******** With points-to analysis ********"

w.activate("proper_effects_with_points_to")
r.privatize_module()
r.coarse_grain_parallelization()
r.display()


