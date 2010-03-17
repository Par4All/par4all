      program w11

C     Assignments in body: check that the initial condition I=0 and the
C     body exit condition 0<=I<=1 are merged as entry condition for the
C     loop body before the fixpoint is computed.

C     Since I may be increasing or decreasing, no information about I is
C     derived in the loop transformer T*

      integer i

      i = 0

      do while(x.gt.0.)
         read *, y
         if(y.gt.0.) then
            i = 0
         else
            i = 1
         endif
      enddo

      print *, i

      end
