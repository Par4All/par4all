      subroutine trusted_ref13(a, n1, n2, m1, m2)

C     Check that nested loops are properly analyzed

      real a(n1, n2)

      do j = 1, m2
         a(i,j) = 0
      enddo

      end
