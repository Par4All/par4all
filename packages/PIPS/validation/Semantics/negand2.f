C     to test negation of and conditions
      
      subroutine negand2
      logical c
      
      if(c) then
         i=1
      else
         i = 2
      endif
      
      if(i.ne.1) then
         j = i
      else
         j = i+1
      endif
      
      i = 4
      print *, j
      
      l = 3
      if(k.eq.1.or.k.eq.2) then
         l = k
      endif
      
      k = 0
      print *, l
      
      if(ii.ge.n) then
         if(ii.eq.n) then
            j = ii
            ii = 0
            print *, j
         else
            j = ii
            ii = 0
            print *, j
         endif
      endif

      print *, i, ii, j, k, l

      end

