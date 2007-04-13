      program side_effect02

C     Check handling of side effects in tests

      i = 4
      j = 0

      if(incr(i).ge.incr(i)) then
         j = 8
      else
         j = 9
      endif

      print *, i, j

c      if(4.lt.incr(i)) then
c         j = 8
c      endif
c
c      print *, i, j

      end

      integer function incr(k)
      save x
      x = x+1.
      k = k + 1
      incr = k
      end
