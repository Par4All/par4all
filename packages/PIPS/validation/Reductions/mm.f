c square matrix multiplication
      program matmul
      integer n
      parameter (n=10)
      real*8 a(n,n), b(n,n), c(n,n)
      integer i, j, k
      do j=1, n
         do i=1, n
            a(i,j) = real(i-(n/2))/real(j)
            b(i,j) = real(j-3)/real(i)
         enddo
      enddo
c
c matrix multiply: C=A*B
c
      do j=1, n
         do i=1, n
            c(i,j) = 0.
            do k=1, n
               c(i,j) = c(i,j) + a(i,k)*b(k,j)
            enddo
         enddo
      enddo
c
c again..
c
      do j=1, n
         do i=1, n
            c(i,j) = 0.
         enddo
      enddo
      do k=1, n
         do j=1, n
            do i=1, n
               c(i,j) = c(i,j) + a(i,k)*b(k,j)
            enddo
         enddo
      enddo
c
c output of the result
c
      print *, ((c(i,j), i=1, n), j=1, n)
      end
!tps$ activate PRINT_CODE_PROPER_REDUCTIONS
!tps$ display PRINTED_FILE
!tps$ activate PRINT_CODE_CUMULATED_REDUCTIONS
!tps$ display PRINTED_FILE
