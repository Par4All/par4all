!
!
!
      program samedist
      integer n
      parameter (n=100)
      real A(n,n)

!hpf$ processors P(2)
!hpf$ template T(n,n)
!hpf$ distribute T(*,block) onto P
!hpf$ align with T(i,j):: A(i,j)

!hpf$ dynamic A

      A(10,10) = 1.

!hpf$ realign with T(*,j):: A(*,j)

      A(11,11) = 2.

      end 

