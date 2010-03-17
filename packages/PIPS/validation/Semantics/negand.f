C     to test negation of and conditions

      subroutine negand

      i=1
      if(i.eq.2.and.j.eq.3) then
         j = 4
      else
         j = 5
      endif

      k = j

      print *, i, j, k

      end
