      subroutine trusted_ref12(a, n1, n2, m1, m2)

C     Check that nested loops are properly analyzed: since you are not
C     sure to enter the i loop, you cannot use any information from the
C     j loop and you end up with nothing

      real a(n1, n2)

      do i = 1, m1
         do j = 1, m2
            a(i,j) = 0
         enddo
C     a comment to have an inner block and see the inner loop
C     transformer
         continue
      enddo

      end
