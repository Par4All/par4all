c kill directive on a array which is also a template
      program dead
      integer n
      parameter (n=10)
      integer A(n,n),i,j
!hpf$ dynamic A
!hpf$ processors P1(4), P2(2,2)
!hpf$ distribute A(block,*) onto P1
      print *, 'in'
!hpf$ independent(j,i)
      do j=1, n
         do i=1, n
            A(i,j) = 1
         enddo
      enddo
      print *, A(1,1)
      print *, 'after init 1'
! should avoid the actual redistribution...
!fcd$ kill A
!hpf$ redistribute A(block,block) onto P2
!hpf$ independent(j,i)
      do j=1, n
         do i=1, n
            A(i,j) = 2
         enddo
      enddo
      print *, A(1,1)
      print *, 'after init 2'
      end
