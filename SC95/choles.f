      program choles
      integer n
      parameter (n=10)
      real*8 a(n,n), p(n), x
      integer i, j, k
      do i=1,n
 1       x = a(i,i)
         do k = 1, i-1
 2          x = x - a(i,k)**2
         end do
 3       p(i) = 1.0/sqrt(x)
         do j = i+1, n
 4          x = a(i,j)
            do k=1,i-1
 5             x = x - a(j,k) * a(i,k)
            end do
 6          a(j,i) = x * p(i)
         end do
      end do
      end
