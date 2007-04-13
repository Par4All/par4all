      subroutine loop01(t, l)

C     Check loop entrance condition

      real t(l)

      i = 2
 
      do i = 1, m*m+2
         t(i) = 0.
      enddo

      print *, i
 
      end
