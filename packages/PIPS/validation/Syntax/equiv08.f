      program equiv08

C     Check that commons have right sizes

      common /foo/x
      real y(100)
      equivalence (x, y)

      common /bar/a, b
      double precision a, b

      do i = 2, 10
         y(i) = 0.
      enddo

      call subequiv08

      end

      subroutine subequiv08

      real y(100)
      equivalence (x, y)
      common /foo/x

      double precision a, b
      common /bar/a, b

      end
