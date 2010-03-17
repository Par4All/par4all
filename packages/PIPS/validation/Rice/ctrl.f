C     Forgotten dependences from statement to control 
C
      program ctrl
      parameter (n=10)
      real a(n,n)
c
      do i = 1, n
         m = i
         do j = 1, m
            a(i,j) = 0.
         enddo
      enddo
      end
