      program loopexit

c     Is it possible to say something about the exit value of 
C     loop indices

      real t(10)

      if(n.ge.1) then

         do i = 1, n, 1
            t(i) = 0.
         enddo

         print *, i

         do i = 1, n, -1
            t(i) = 0.
         enddo

         print *, i

      endif 

      if(n.ge.0) then

         print *, i
         do i = 1, n, 1
            t(i) = 0.
         enddo

         print *, i

         do i = 1, n, -1
            t(i) = 0.
         enddo

         print *, i

      endif

      if(n.le.1) then

         print *, i
         do i = 1, n, 1
            t(i) = 0.
         enddo

         print *, i

         do i = 1, n, -1
            t(i) = 0.
         enddo

         print *, i

      endif

      do i = 1, n, 1
         t(i) = 0.
      enddo

      print *, i

      do i = 1, n, -1
         t(i) = 0.
      enddo

      end
