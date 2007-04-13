      subroutine trusted_ref10(a, n)

      real a(n, n)

      do i = 1, n
         do j = 1, n
            a(i,j) = 0
         enddo
      enddo

      end
