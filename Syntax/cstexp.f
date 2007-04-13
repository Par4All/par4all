      program cstexp
c     To check that max0
      parameter (i=1, j=2*i, k = max0(i,j))
      real a(i), b(j), c(k)

      print *, a(i)

      l = k

      print *,l

      end
