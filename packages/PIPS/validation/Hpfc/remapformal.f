      program formalremappednotused
      integer n
      parameter (n=10)

!hpf$ processors P(2)
!hpf$ template T(n)
!hpf$ distribute T(block) onto P
      
      real sol(n)
!hpf$ align with T:: sol
      
      sol(1)=1
      call noremap(sol)
      end

      subroutine noremap(sol)
      integer n
      parameter (n=10)

!hpf$ processors P(2)
!hpf$ template T(n)
!hpf$ distribute T(block) onto P
      
      real sol(n)
!hpf$ align with T:: sol

      call remap(sol)
      sol(1)=2
      end

      subroutine remap(sol)
      integer n
      parameter (n=10)

!hpf$ processors P(2)
!hpf$ template T(n)
!hpf$ distribute T(block) onto P
      
      real sol(n), s
!hpf$ align with T(*):: sol(*)

      s=sol(1)+sol(2)
      end
