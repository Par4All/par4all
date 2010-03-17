      subroutine trusted_ref14(a, n1, n2, m1, m2)

C     Check that nested loops are properly analyzed: since you are 
C     sure to enter the i loop, you can use information from the
C     j loop and you end up with everything (see trusted_ref12.f)

      real a(n1, n2)

      if(m1.lt.1) stop

      do i = 1, m1
         do j = 1, m2
            a(i,j) = 0
         enddo
C     a comment to have an inner block and see the inner loop
C     transformer
         continue
      enddo

      end
