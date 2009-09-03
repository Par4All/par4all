c square matrix multiplication
      program matmul2
      integer n
      parameter (n=1000)
      integer a(n,n), b(n,n), c(n,n), x
      integer i, j, k
C initialize the square matrices with ones
      do j=1, n
         do i=1, n
            a(i,j) = 1
            b(i,j) = 1
         enddo
      enddo
C multiply the two square matrices of ones
      do j=1, n
         do i=1, n
            x = 0
            do k=1, n
               x = x + a(i,k)*b(k,j)
            enddo
            c(i,j) = x
         enddo
      enddo
C The result should be a square matrice of n
      do j=1, n
         do i=1, n
            if (c(i,j) .NE. n) then
              print *, ("CHECK FAILED")
              goto 100
           endif
         enddo
      enddo
      print *, ("CHECK SUCCEED")
 100  end

