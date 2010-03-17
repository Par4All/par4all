      PROGRAM W06

C     Check fixpoint on conditional incrementation

      INTEGER I, N
      REAL T(10)

c      I = 100
      I = 1
      N = 10
      J = 3

      DO WHILE (I.LT.N)
         PRINT *, I 
         IF(T(I).GT.0.) THEN
            J = J-2
         ENDIF
         I = I+1
      ENDDO

      PRINT *, I, J

      END
