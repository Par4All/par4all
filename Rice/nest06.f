!     Example p.4 Khadija Imadoueddine: matrix multiply
!     
!     The k loop must be moved inside

!     But it is not parallel, which PIPS does not detect (wrong code is
!     generated) This should lead to some register tiling
!
!     Legality of interchange is not ested either

      subroutine interchange(n, a, b, c)
      integer n, i, j, k
      real a(1:n,1:n), b(1:n,1:n), c(1:n,1:n)

      do 300 k = 1, n
         do 200 j = 1, n
            do 100 i = 1, n
               c(i,j) = c(i,j) + a(i,k)*b(k,j)
 100        continue
 200     continue
 300  continue

      end
