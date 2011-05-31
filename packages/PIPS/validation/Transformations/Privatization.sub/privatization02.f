Check that J is privatized in the outermost loop

      subroutine privatization02
      integer a(10)

      do i = 1, 10
         j = 0
         do k = 1, 10
            j = j + 1
         enddo
         a(i) = j
      enddo
      end
