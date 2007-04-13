      program w08

C     The evolution of ms cannot be tracked because there are no other
C     variables? The loop transformer is not precise because dms.ge.-2
C     is inconclusive about dms evolution (it may be increasing at any
C     rate or decreasing no faster than -2 per iteration), but the
C     preconditions are precise in the loop and at loop exit.

      integer ms

      ms = 0

      do while(ms.le.2)
         if(x.gt.0.) then
            ms = 0
         else
            ms = ms + 1
         endif
      enddo

      print *, ms

      end
