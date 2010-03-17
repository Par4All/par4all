C     Check that J is not privatized in the first loop, but privatized
C     in the second one. This is the same vase as privatization01.f, but
C     with a proper initialization of J

      subroutine privatization08
      integer a(10)

      j = 0
      do i = 1, 10
         j = j + 1
         a(i) = j
      enddo

      do i = 1, 10
         j = 1
         a(i) = j
      enddo
      end
