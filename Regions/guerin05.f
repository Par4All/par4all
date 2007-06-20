      program guerin05

C     Bugs in IN and OUT scalar regions for the specified sections:
C     scalar k is not seen as imported by the loop, although it is
C     imported by the loop body. But, if k is replaced by k(1), k(1) is
C     seen as simported.

      implicit none

      integer k, i

      k = 0
      do i = 1, 1
         k = k + 1
      enddo

      end
