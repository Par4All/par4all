C     Check that j is not privatized in the first loop, but privatized
C     in the second one... Here j is imported from the caller. Its
C     storage is not compatible with a privatzation.

C     See also privatization01, 07, 08, 09 and 10

      subroutine privatization10(j)
      integer a(10)

      do i = 1, 10
         j = j + 1
         a(i) = j
      enddo

      do i = 1, 10
         j = 1
         a(i) = j
      enddo
      end
