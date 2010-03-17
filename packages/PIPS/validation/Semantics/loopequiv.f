      subroutine loopequiv(N, T)
      integer N, M, K
      real T(100), X

C     Test: there should be no information about the loop
C     indices because the upper loop bound cannot be analyzed
C     due to an EQUIVALENCE, and/or the loop body modifies 
C     directly or indirectly the value of the upper bound expression

      equivalence (M,X), (K,L)

      do I = I+1, M
         T(I)=0.
      enddo

      do I = I+1, N+L
         K=2
         T(I)=0.
      enddo

      do I = I+1, N
         N=0
         T(I)=0.
      enddo

      print *, I, N ,M , K, L
      end
