      program distribution
      real A(100,100), B(100,200)
      real C(100,100), D(100,100), E(100,100)
!hpf$ align (i,j) with E(j,i) :: C, D
!hpf$ processors P(2,2)
!hpf$ distribute (block,block) onto P :: A, B, E
      end
