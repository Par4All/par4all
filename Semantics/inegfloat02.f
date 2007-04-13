      program inegfloat02
      real a, b
      read *, a, b
      if (a+b.eq.0.0) then
         print *, 'a+b = 0'
         print *, a+b
      endif
      if (a.gt.0.0.and.b.gt.0.0) then
         print *, 'a>0 and b>0'
         print *, a+b
         if (a+b.gt.0.0) then
            print *, 'test redondant a+b>0'
            print *, a+b
         endif
      endif
      if (a.eq.b) then
         print *, 'a-b = 0'
         print *, a-b
      endif
      if (a.eq.-b) then
         print *, 'a+b = 0'
         print *, a+b
      endif
      end
