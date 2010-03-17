! test moving bounds...
      program movebounds
      integer n
      parameter (n=100)

      real A(n,n)
      integer i,j,k

!hpf$ processors P(2,2)
!hpf$ distribute A(block,block) onto P

      do k=1, n
!hpf$    independent
         do j=1, k
!hpf$       independent
            do i=n-10, n
               A(i,j) = A(i,j) + k + j + i
            enddo
         enddo
      enddo

      end
