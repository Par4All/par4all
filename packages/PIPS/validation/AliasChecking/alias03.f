C     example with EQUIVALENCE

      PROGRAM ALIAS
      INTEGER A(2,3), B(3),C(5),SUB
      EQUIVALENCE (SUB,A(1,2),B(2),C(3))
      EQUIVALENCE (A(1,2),B(2),C(3))
      EQUIVALENCE (M,N)
      EQUIVALENCE (M,K)
      COMMON W
      CALL FOO(M,N)
      M = 5
      A(1,N) = A(2,N) + A(1,N)
      B(2) = 3
      READ *,A
      CALL FOO(W,SUB)
      CALL FOO(A(1,2),B(2))
      END

      SUBROUTINE FOO(NV1,NV2)
      COMMON W
      NV1 = 1
      NV2 = 2
      END
      
