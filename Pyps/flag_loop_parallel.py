from validation import vworkspace

with vworkspace() as w:
  f=w.fun.flagloopexecution
  print "Loop Parallel?",f.loops("loop_lab").parallel
  f.loops("loop_lab").parallel=True
  l=f.loops("loop_lab")
  l.parallel=True
  print "Loop Parallel?",l.parallel, f.loops("loop_lab").parallel
  print '# Loop has been flag as parallel'
  f.display()

  f.loops("loop_lab").parallel=False
  print '# Loop has been flag as sequential'
  f.display()

