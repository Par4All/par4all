      program loopexit6

c     Check non-unit increments, the loop may be entered or not

      real t(10)

      do i = 1, n, 2
         t(i) = 0.
      enddo

C     Expected precondition:
C     P(I) {N+1<=I} 
      print *, i

      end
