      subroutine loopequiv2(N, T)
      integer N, M, K
      real T(100), X

C     Test: there should be no information about the loop
C     indices because the loop increment cannot be analyzed
C     due to an EQUIVALENCE, and/or the loop body modifies 
C     directly or indirectly the value of the increment which 
c     should not matter because it does not appear explictly
C     in preconditions (only its sign at loop entry matters).

      equivalence (M,X), (K,L)

      M = 1

C     The increment sign is unknown because M is not analyzed

      do I = I+1, N, M
         T(I)=0.
      enddo

      L = 1

C     Although L is implictly modified in the loop body, the increment sign
C     and value are known at loop entry.

      do I = I+1, N, L
         K=2
         T(I)=0.
      enddo

      N = 1

C     Although N is modified in the loop body, the increment sign
C     and value are known at loop entry.

      do I = I+1, L, N
         N=0
         T(I)=0.
      enddo

      print *, I, N ,M , K, L

      end
