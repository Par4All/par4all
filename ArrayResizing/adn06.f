      PROGRAM INTER_REGION
      REAL WORK(10)
      I = 2
      N = 2 + FOO(5)
      K =1
      M =2
      CALL RUN(WORK)
      L =1
      IF (K.GT.FOO(4)) THEN
         CALL RUN(WORK)
      ELSE
         M = FOO(3)
      ENDIF
     
      H =1 
      END
      SUBROUTINE RUN(C)
      REAL C(1)
      DO 10 I=1,20
         C(I)=0.
 10   CONTINUE
      END
      FUNCTION FOO(I)
      FOO = I
      END
