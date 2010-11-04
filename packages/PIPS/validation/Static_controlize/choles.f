      program choles
      integer n
      real*8 a(n,n), p(n), x
      integer i, j, k
      do i=1,n
         x = a(i,i)
         do k = 1, i-1
            x = x - a(i,k)**2
         end do
         p(i) = 1.0/sqrt(x)
         do j = i+1, n
            x = a(i,j)
            do k=1,i-1
               x = x - a(j,k) * a(i,k)
            end do
            a(j,i) = x * p(i)
         end do
      end do
      end
      
