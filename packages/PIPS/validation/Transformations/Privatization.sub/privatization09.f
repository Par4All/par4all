C     Check that j is not privatized in the first loop, but privatized
C     in the second one... However, j is not initialized before its use
C     in the first loop, the value of j does not matter and hence j can
C     be privatized in both loops, But unlike privatization01,f, j is
C     used in the next loop nest. Hence j cannot be privatized.

      subroutine privatization09
      integer a(10)

      do i = 1, 10
         j = j + 1
         a(i) = j
      enddo

      do i = 1, 10
         a(i) = j
      enddo
      end
