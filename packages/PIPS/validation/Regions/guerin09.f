      program guerin09

C     Bugs in IN and OUT scalar regions for the specified sections:

      implicit none

      integer k(10), i

      k(1) = 0
      do i = 2, 10
         k(i) = k(i-1)+1
      enddo

      end
