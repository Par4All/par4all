      program killandremap
      integer A(10)

!hpf$ processors P(2)
!hpf$ distribute A(block) onto P
!hpf$ dynamic A

      A(1) = 1

!fcd$ kill A
!hpf$ redistribute A(block(8)) onto P

      A(1) = 2

!hpf$ redistribute A(block) onto P

      A(1) = 3

      end

