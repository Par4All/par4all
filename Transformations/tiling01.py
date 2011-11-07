from validation import vworkspace

with vworkspace() as w:
  w.props.memory_effects_only = False
  w.activate('must_regions')
  tiling_vector=["N"]
  m = w.fun.test
  m.loop_normalize()
  m.internalize_parallel_code()
  for l in m.loops():
    m.display()
    if l.loops():
      try:
        l.symbolic_tiling(force=True,vector=",".join(tiling_vector))
	l.set_execution_parallel(False)
      except:
        raise

  m.loop_normalize(one_increment=True)
  m.display()

