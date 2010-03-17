C     test de la (non) capture des intrinsics, e.g. DIM
      program capture01
      dim = 1
      if(dim.lt.0.) print *, dim
      end
