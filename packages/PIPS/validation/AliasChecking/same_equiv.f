      PROGRAM MAIN
      REAL V1(20,20),V2(100)
      EQUIVALENCE (V1(1,1),V2(5))
      EQUIVALENCE (U,V)
      CALL FOO(V1,A,B,C,B,B,V1(I,J),V2(K),C,A,M,U,V)
      END
      SUBROUTINE FOO(F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13)
      REAL F1(20,20),F7(10,10),F8(50)
      F1(1,1) =1
      F2 = 1
      F3 = 1
      F4 = 1
      F5 = 1
      F6 = 1
      F7(2,2) = 1
      F8(1) = 1
      F9 = 1
      F10 = 1
      F11 = 1
      F12 = 1
      F13 =1
      END
