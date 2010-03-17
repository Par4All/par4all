      program ovfl01

      i = 2**30
      j = i - 1
      k = i + j
      l = k + 1
      print *, l

      if(l.ge.0) then
         print *, 'l is positive'
      else
         print *, 'l is strictly negative'
      endif

      end
