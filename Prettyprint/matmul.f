c square matrix multiplication
      program matmul
      integer n
      parameter (n=10)
      real*8 a(n,n), b(n,n), c(n,n), x
      integer i, j, k
      do j=1, n
         do i=1, n
            a(i,j) = real(i-(n/2))/real(j)
            b(i,j) = real(j-3)/real(i)
         enddo
      enddo
      do j=1, n
         do i=1, n
            c(i,j) = 0.
            do k=1, n
               c(i,j) = c(i,j) + a(i,k)*b(k,j)
            enddo
         enddo
      enddo
      do j=1, n
         do i=1, n
            x = 0.
            do k=1, n
               x = x + a(i,k)*b(k,j)
            enddo
            c(i,j) = x
         enddo
      enddo
      print *, ((c(i,j), i=1, n), j=1, n)
      end
