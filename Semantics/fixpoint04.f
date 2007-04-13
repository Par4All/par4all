      program fixpoint04

C     Information about j should not be lost because the convex closure of j
C     unchanged and j = 5 is constrained by condition j >= 5

C     But it does not work with the derivative fixpoint because the
C     condition j <= 5 in the loop body transformer is lost by that
C     fixpoint operator and legaly so since the loop may not be entered

C     The loop may not be entered: no information about j is available
C     when it is printed

      do i = 1, n
         if(j.ge.5) then
            j = 5
         endif
      enddo

      print *, j

C     The loop is entered: the fix point is not improved (alas) but the
C     loop body is applied at least once which produces a condition on
C     J, namely J <= 5

      do i = 1, 5
         if(j.ge.5) then
            j = 5
         endif
      enddo

      print *, j

C     Suppose J is known to be greater than 5 (in fact, equal to 5
C     because of the previous loop) and the loop certainly entered: the
C     information is lost!

      if(j.ge.5) then

         do i = 1, 5
            if(j.ge.5) then
               j = 5
            endif
         enddo

         print *, j

      endif

      end
