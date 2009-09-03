c square matrix multiplication
      program matmul1
      integer n
      parameter (n=1000)
      integer a(n,n), b(n,n), c(n,n), x
      integer i, j, k
      do j=1, n
         do i=1, n
            a(i,j) = 1
            b(i,j) = 1
         enddo
      enddo
      do j=1, n
         do i=1, n
            c(i,j) = 0
            do k=1, n
               c(i,j) = c(i,j) + a(i,k)*b(k,j)
            enddo
         enddo
      enddo
      print *, ((c(i,j), i=1, n), j=1, n)
      end
