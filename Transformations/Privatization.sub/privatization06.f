C     Check that j is privatized when there is a def-def arc

      subroutine privatization06
      integer a(10, 10)

      do i = 1, 10
         do k = 1, 10
            j = 0
            j = 1
            a(i, k) = j
         enddo
      enddo
      end
