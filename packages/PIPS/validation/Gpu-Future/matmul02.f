c square matrix multiplication
      program matmul02
      integer n
      parameter (n=100)
      integer a(n,n), b(n,n), c(n,n), x
      integer i, j, k
      logical flg
      print *,n
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
      flg = .TRUE.
      do j=1, n
         do i=1, n
            flg = flg .AND. (c(i,j) .EQ. n)
         enddo
      enddo
      if (flg .EQV. .FALSE.) then
         print *, ("CHECK FAILED")
      else
         print *, ("CHECK SUCCEED")
      endif
      end

