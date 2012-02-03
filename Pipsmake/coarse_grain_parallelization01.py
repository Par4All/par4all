from validation import vworkspace
import os,re

with vworkspace() as w:
  w.activate("must_regions")
  # We should be able to parallelize anything here
  w.all_functions.coarse_grain_parallelization()
  w.all_functions.display()
  
  # CODE resource shouldn't have been changed, then no recompute is required
  # for EFFECTS, PRECONDITIONS, REGIONS, etc.
  w.all_functions.coarse_grain_parallelization()
  
  # Print phase activated, check that useless dependance weren't recomputed
  filename = os.path.join(w.dirname,'Logfile')
  with open(filename, 'r') as f:
    for l in f:
      if re.search('(building|updating)', l) != None:
         print l,
  
