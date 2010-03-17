C     to test or conditions
      
      subroutine or
      
      l = 3
      if(k.eq.1.or.k.eq.2) then
      l = k
      endif
      k = m
      print *, l
      
      l = 3
      if(k.ge.1) then
         if(k.eq.1.or.k.eq.2) then
            l = k
            k = 0
            print *, l
         else
            l = k
            k = 0
            print *, l
         endif
      endif

      print *, k, l, m
      
      end

