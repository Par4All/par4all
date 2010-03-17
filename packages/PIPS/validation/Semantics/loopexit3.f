      program loopexit3

c     The postcondition of a non-executed loop should be its precondition 
c     plus the index initialization

      real t(10)

      if(n.lt.1) then

         j = 0
         do i = 1, n, 1
            t(i) = 0.
            j = j + 2
         enddo

         print *, i, j

      endif

      end
