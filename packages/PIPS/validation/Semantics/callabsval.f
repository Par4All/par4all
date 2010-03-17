      program callabsval

c     bug: results are better in absval than in callabsval because
c     the absval transformer does not take the test condition into account

c     note also that preconditions in the test branches are not symmetrical
c     because 0 belongs to the [-1..1] interval and because the condition
c     .lt. is more precise when n is negative

      i = 1
      call absval(i, iabs)
      i = -1
      call absval(i, iabs)
      print *, i, iabs
      end
      subroutine absval(n, nabs)
      if(n.lt.0) then
         nabs = -n
      else
         nabs = n
      endif
      end
