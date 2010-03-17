      PROGRAM INTER_REGION
      REAL WORK(100)
      
      CALL RUN2(WORK)
      
      CALL RUN3(WORK)
      CALL RUN1(WORK)
      END
      SUBROUTINE RUN1(C)
      COMMON /FOO/ N
      REAL C(1)
      CALL RUN(C)
      END
      SUBROUTINE RUN2(C)
      COMMON /FOO/ N
      REAL C(1)
      CALL RUN(C)
      END
      SUBROUTINE RUN3(C)
      COMMON /FOO/ N
      REAL C(1)
      CALL RUN(C)
      END
      SUBROUTINE RUN(C)
      COMMON /FOO/ N
      REAL C(1)
      DO 10 I=1,N
         C(I)=0.
 10   CONTINUE
      END
