C     Double use of x for a formal parameter ends up with an abort instead of a Parser Error

      subroutine scalarization35(x, y, x, n)
      real x(n), y(n), z(n)

      do i = 2, n
         x(i) = z(i)*(y(i)-x(i-1))
      enddo

      print *, x

      end

