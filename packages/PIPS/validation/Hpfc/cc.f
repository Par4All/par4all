! test with processor lattice.
      program cc
      integer n, i, j
      parameter (n=100)
      real a(n,n)
!hpf$ dynamic a
!hpf$ template t(n,n)
!hpf$ processors p(2,2)
!hpf$ align with t:: a
!hpf$ distribute t(cyclic,cyclic) onto p

      a(1,1) = 1.0

!hpf$ realign a(i,j) with t(j,i)

      print *, a(1,1)

      end
