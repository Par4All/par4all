      program matadd
      integer m
      integer n
      parameter (m = 402, n = 160)
      integer i, j
      real a(m, n), b(m, n), c(m, n)

      do i=1,m
         do j=1,n
            b(i,j) = i
            c(i,j) = j
         end do
      end do

      do i=1,m
         do j=1,n
            a(i,j) = b(i,j) + c(i,j)
         end do
      end do

      end
