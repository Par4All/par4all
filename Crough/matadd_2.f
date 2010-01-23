! This is mainly to test that a random lower/uppercase works between
! Fortran and C with CROGH called from C_PRINTED_FILE
      program matadd
      integer m
      integer N
      parameter (M = 402, n = 160)
      integer i, J
      real a(M, n), b(m, N), c(m, n)

      do i=1,M
         do J=1,n
            B(i,j) = i
            c(i,J) = j
         end do
      end do

      do I=1,m
         do j=1,N
            A(I,j) = b(i,J) + C(i,j)
         end do
      end do

      end
