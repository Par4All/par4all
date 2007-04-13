      subroutine redeclaredim
      common /foo/ x(10)
      real x(3)

      do i = 1, 10
         x(i) = 0.
      enddo

      end
