      subroutine trusted_ref11(a, n)

C     Check that the inner loop is analyzed as always entered since the
C     outer loop implies m>=1

      real a(n, n)

      read *, m

      do i = 1, m
         do j = 1, m
            a(i,j) = 0
         enddo
         print *, j
      enddo

      end
