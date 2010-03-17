      program loopinit3

c     Check that loop index initialization is performed correctly: the initial
c     value of i is preserved by the loop because the loop is never executed.

c     Simplified version of loopinit2 because inc(j) appears in an assignment
c     instead of a loop lower bound.

      real t(10)

      j = 2

      n = 0

      k = inc(j)

      do i = k, n, 1
         t(i) = 0.
         j = j + 2
      enddo

      print *, i, j

      end
      
      integer function inc(k)
      k = k + 1
      inc = k
      end
