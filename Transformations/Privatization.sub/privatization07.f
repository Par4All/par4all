C     Check that J is not privatized in the outermost loop very similar
C     to privatization02.f)

      subroutine privatization07
      integer a(10)

      j = 0
      do i = 1, 10
         do k = 1, 10
            j = j + 1
         enddo
         a(i) = j
      enddo
      end
