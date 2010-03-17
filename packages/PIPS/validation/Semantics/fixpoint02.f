      program fixpoint02

C     Check that constant assignments are taken into account when loop
C     may be entered or not but when initial values are known

      j = 3

      do i = 1, n
         j = 4
      enddo

      print *, j

      do i = 1, n
         if(x.gt.0.) then
            j = 5
         else
            j = 6
         endif
      enddo

      print *, j

      end
