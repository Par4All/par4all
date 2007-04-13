      program trusted_ref17

C     Check that non-executable loop bodies generate conditions for loop bounds

      real a(1)

c      read *, n

      do i = 1, n
         a(2) = 0.
         continue
      enddo

      print *, n

      end
