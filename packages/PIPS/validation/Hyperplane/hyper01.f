      subroutine hyper01(x, n, m)

      real x(n, m)

      do 100 i = 1, n
         do 200 j = 1, m
            x(i,j) = float(i+j)
 200     continue
 100  continue

      print *, x

      end
