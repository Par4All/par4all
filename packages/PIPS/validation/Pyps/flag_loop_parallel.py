from validation import vworkspace

with vworkspace() as w:
  f=w.fun.flagloopexecution
  f.loops("loop_lab").parallel(True)
  print '# Loop has been flag as parallel'
  f.display()

  f.loops("loop_lab").parallel(False)
  print '# Loop has been flag as sequential'
  f.display()

