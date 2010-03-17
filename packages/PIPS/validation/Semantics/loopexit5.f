      program loopexit5

C     Propagate information although it is not known if the loop is executed
C     or not:
C     this Requires a convex hull of the loop entered and 
C     the loop non entered cases

      real t(10)

      i = 0
      do i = n, 1, -1
         t(i) = 0.
      enddo

C     Expected but not found because FALSE precondition:
C     P(I) {I==0} 
C     Effective precondition because initialization of i is useless
C     and because n's value is unknown:
C     P(I) {I<=0}

      print *, i

      end
