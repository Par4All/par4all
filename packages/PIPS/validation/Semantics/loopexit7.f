      program loopexit7

c     Check non-unit increments:
C     the loop is entered, increment is not 1

      real t(10)

      if(n.ge.1) then
         do i = 1, n, 2
            t(i) = 0.
         enddo

C        Expected precondition:
C        P(I) {N+1<=I, I<=N+2, 1<=N} 
         print *, i

      endif

      end
