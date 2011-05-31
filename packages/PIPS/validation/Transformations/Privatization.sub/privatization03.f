C     Check that j is privatized

      subroutine privatization03
      integer a(10)

      do i = 1, 10
         j = 0
         j = j + 1
         a(i) = j
      enddo
      end
