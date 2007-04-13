      program loopexit2

c     Is it possible to say something about the exit value of 
C     loop indices

      real t(10)

      if(n.ge.1) then

         j = 0
         do i = 1, n, 1
            t(i) = 0.
            j = j + 2
         enddo

         print *, i, j

         j = 0
         do i = 1, n, -1
            t(i) = 0.
            j = j + 2
         enddo

         print *, i, j

      endif 

      if(n.ge.0) then

         j = 0
         do i = 1, n, 1
            t(i) = 0.
            j = j + 2
         enddo

         print *, i, j

      endif

      if(n.lt.1) then

         j = 0
         do i = 1, n, 1
            t(i) = 0.
            j = j + 2
         enddo

         print *, i, j

      endif

      j = 0
      do i = 1, n, 1
         t(i) = 0.
         j = j + 2
      enddo

      print *, i, j

      end
