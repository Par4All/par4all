      PROGRAM MAIN
      INTEGER IC, JC, KC
      COMMON/IJKC/ IC,JC,KC
      INTEGER IM1, JM1, KM1
      COMMON/LI/ IM1,JM1,KM1
      REAL F1(10),F2(10)
      IC = 3
      JC = 3
      KC = 3
      CALL FONT3(F1,F2)
      PRINT *,"What ?"
      END
      SUBROUTINE  FONT3(F1 , F2)
      INTEGER IC, JC, KC
      COMMON/IJKC/ IC,JC,KC
      INTEGER IM1, JM1, KM1
      COMMON/LI/ IM1,JM1,KM1
      REAL    F1, F2
      DIMENSION F1(IC,JC,1),F2(IC,JC,1)
      INTEGER I, J

      DO 15 I = 1, IC                                                   0001
         DO 15 J = 1, JC                                                0002
            DO 15 K = 1,1                                               0003
               F2(I,J,K) = F1(I,J,K)                                    0004
15             CONTINUE                                                 0005
      END
