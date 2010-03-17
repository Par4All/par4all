      program fixpoint01

C     Check that constant assignments are taken into account when loops
C     are certainly entered

      do i = 1, 10
         j = 4
      enddo

      print *, j

      do i = 1, 10
         if(x.gt.0.) then
            j = 4
         else
            j = 5
         endif
      enddo

      print *, j

      end
