C     Check that j is not privatized in the first loop, but privatized
C     in the second one... However, j is not initialized before its use
C     in the first loop, the value of j does not matter and hence j can
C     be privatized in both loops. But j is statically initialized by a
C     DATA and allocated in the STATIC area, which makes it non eligible
C     for privatization.

C     See also privatization01, 07, 08, 09 and 10

      subroutine privatization11
      integer a(10)
      data j/0/

      do i = 1, 10
         j = j + 1
         a(i) = j
      enddo

      do i = 1, 10
         j = 1
         a(i) = j
      enddo
      end
