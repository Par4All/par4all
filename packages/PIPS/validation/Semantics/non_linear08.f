      subroutine non_linear08(t, l)

C     Check rules for multiplication and loop bounds

      real t(l)

      read *, n
      do i = n*n+2, m
         t(i) = 0.
      enddo

      print *, i
 
      do i = 1, m*m+2
         t(i) = 0.
      enddo

      print *, i
 
      end
