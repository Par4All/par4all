      subroutine hyper02(x, n, m)

C     Same code as loopfi01.f, used to check management of user errors

      real x(n, m)

      do 100 i = 1, n
         do 200 j = 1, m
            x(i,j) = float(i+j)
 200     continue
 100  continue

      print *, x

      end
