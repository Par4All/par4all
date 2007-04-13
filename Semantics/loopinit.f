      program loopinit

c     Check that loop index initialization is performed correctly

      real t(10)

      j = 2

      do i = inc(j), n, 1
         t(i) = 0.
         j = j + 2
      enddo

      print *, i, j

      end
      
      integer function inc(k)
      k = k + 1
      inc = k
      end
