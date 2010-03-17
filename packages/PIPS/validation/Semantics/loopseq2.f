      program loopseq2

C     Check exit condition from first loop onto second loop

      real t(100)

      if(n.ge.1) then

         do i = 1, n
            t(i) = 0.
         enddo

         do j = i, n
            t(j) = 0.
         enddo

      print *, i, j

      endif

      end
