      program loopexit8

c     Check exit post-condition: 
C     the loop is entered and increment is 1

      real t(10)

      if(n.ge.1) then
         do i = 1, n
            j = i+mmax
            t(i) = 0.
         enddo

C        Expected precondition:
C        P(I) {I+MMAX==J+1, N+1<=I, I<=N+1, 1<=N} 
         print *, i, j, mmax

      endif

      end
