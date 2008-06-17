c kill directive on an array
      program dead
      integer n
      parameter (n=10)
      integer A(n,n),i,j
!hpf$ template T(n,n)
!hpf$ dynamic A, T
!hpf$ align with T :: A
!hpf$ processors P1(4), P2(2,2)
!hpf$ distribute T(block,*) onto P1
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
!hpf$ redistribute T(block,block) onto P2
!hpf$ independent(j,i)
      do j=1, n
         do i=1, n
            A(i,j) = 2
         enddo
      enddo
      print *, A(1,1)
      print *, 'after init 2'
      end
