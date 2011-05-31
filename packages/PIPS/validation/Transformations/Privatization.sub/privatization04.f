C     Check that j is not privatized; just likre privatization03.f, but
C     without the initialization. But the lack of initialization makes
C     privatization fine as the output is undefined.

      subroutine privatization04
      integer a(10)

      do i = 1, 10
         j = j + 1
         a(i) = j
      enddo
      end
