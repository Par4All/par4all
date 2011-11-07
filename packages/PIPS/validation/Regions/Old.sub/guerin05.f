      program guerin05

C     Bug fixed in IN scalar regions for the specified sections:
C     scalar k is now correctly seen as imported by the loop.

      implicit none

      integer k, i

      k = 0
      do i = 1, 1
         k = k + 1
      enddo

      print *,k

      end
