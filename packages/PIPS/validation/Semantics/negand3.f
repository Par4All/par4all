C     to test negation of and conditions: subset of negand2
      
      subroutine negand3
      
      l = 3
      if(k.eq.1.or.k.eq.2) then
         l = k
      endif

      print *, k, l

      end

