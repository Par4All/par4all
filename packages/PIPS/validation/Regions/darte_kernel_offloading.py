from validation import vworkspace

with vworkspace() as w:
  w.props.memory_effects_only = False
  w.props.prettyprint_memory_effects_only = True
  w.activate('must_regions')

  fs=w.fun
  for f in [fs.orig,fs.skewed,fs.tiled]:
    f.display('print_code_preconditions')
    f.display('print_code_regions')
    print "***** COARSE GRAIN ******"
    f.privatize_module()
    f.coarse_grain_parallelization()
    f.display()
    print "***** FINE GRAIN ******"
    f.internalize_parallel_code()
    f.display()
  

