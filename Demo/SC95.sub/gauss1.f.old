      program gauss
      integer n
      parameter (n=10)
      real*8 a(n,n), x(n), s, f
      integer i, j, k
      do i = 1,n-1
        do j = i+1,n
           do k = i+1,n
 5            f = a(i,k)/a(i,i)
 3            a(j,k)=a(j,k) - a(j,i)*f
           end do
        end do
      end do
      do i = 1,n
 2       s = 0.
         do j = 1,i-1
 1          s = s + a(n-i+1,n-j+1)*x(n-j+1)
         end do
 4       x(n-i+1) = (a(n-i+1, n+1) - s)
     &        /a(n-i+1, n-i+1)
      end do
      end

