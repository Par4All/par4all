      program loopexit4

c     The postcondition of a certainly executed loop should be its invariant 
c     updated by the last increment plus the exit condition

      real t(10)

      read *, n

      if(n.ge.1) then

         j = 0
         do i = 1, n, 1
            t(i) = 0.
            j = j + 2
            k = 3
         enddo

C        Expected precondition:
C        P(I,J,K,N) {J==2I-2, K==3, N+1<=I, I<=N+1, 1<=N}
         print *, i, j, k

      endif

      end
