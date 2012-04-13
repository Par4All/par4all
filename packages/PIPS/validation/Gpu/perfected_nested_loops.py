from validation import vworkspace

with vworkspace() as w:
  m=w.fun.main
  m.loops()[0].parallel=True
  m.loops()[0].loops()[0].parallel=True  
  
  m.display()
  
  m.gpu_ify(GPU_USE_WRAPPER = False, 
            GPU_USE_KERNEL = False,                             
            GPU_USE_LAUNCHER = True,
            OUTLINE_WRITTEN_SCALAR_BY_REFERENCE = False, # unsure
            annotate_loop_nests = True)
  
  w.all_functions.display()
  
