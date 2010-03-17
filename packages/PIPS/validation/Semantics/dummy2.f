      subroutine dummy2

c     Check translation mechanism when variables are not visible in the
c     current procedure.

      common /foo/ndummy(2),k,l,m,n
      real n

      call bar

      print *, k, l, m, n

      end

      subroutine bar
      common /foo/i,j,k,m,l,n
      i = 1
      j = 2
      k = 3
      l = 4
      m = 5
      n = 6
      end
