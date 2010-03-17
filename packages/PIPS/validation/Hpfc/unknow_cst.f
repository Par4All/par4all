      program cst

      integer n,l,i,j
      parameter (n=10)

      real a(n,n)

!hpf$ processors P(2,2)
!hpf$ distribute A(block,block) onto P

!!hpf$ independent(j,i)
!      do j=1,n
!         do i=1,n
!            a(i,j) = 0.
!         end do
!      end do


      read *, l

!hpf$ independent
      do i=1, n
         a(i,l) = 1.
      end do
      
      end
