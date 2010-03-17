      program fixpoint03

C     Information about j is lost because the convex closure of j
C     unchanged and j = 5 is j can have any value

      j = 4

      do i = 1, n
         if(x.gt.0.) then
            j = 5
         endif
      enddo

      print *, j

      do i = 1, n
         if(j.ge.5) then
            j = 5
         endif
      enddo

      print *, j

      end
