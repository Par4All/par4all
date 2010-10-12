!     Example p. 6 Khadija Imadoueddine: loop distribution, modified by
!     forward substitution of c(i)
!     
!     The loops must be distributed and then exchanged

      subroutine distribution(n1, n2, a, b)
      integer n1, n2, i, j
      real a(1:n1,1:n2), b(1:n1,1:n2), c(n1), d(n1)

      do i = 1, n1
         c(i) = i
         d(i) = i
         do j = 1, n2
            a(i,j) = a(i,j)+b(i,j)*i
         enddo
      enddo

      end
