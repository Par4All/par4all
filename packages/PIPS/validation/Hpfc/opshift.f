      program foo
      integer i,n,m
      parameter(n=100)
      real A(n,n)
!hpf$ processors P(2,2)
!hpf$ distribute A(block,block) onto P

      read *, m

!hpf$ independent 
      do i=1, n
         a(i,m) = -a(i,m-1)
      enddo

      end

      
