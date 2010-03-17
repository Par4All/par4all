      program loopexit9

C     Check than n < 1 after the loop if no array overflow occurs

      real u(1)

      do i = 1, N
         u(i+1) = 0.
      enddo

      print *, n

      end
