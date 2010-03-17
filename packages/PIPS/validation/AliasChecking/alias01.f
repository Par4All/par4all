      PROGRAM MAIN
      REAL WORK(2000)
      READ *,N
      CALL RUN(WORK,WORK(N+1),N)
      END
      SUBROUTINE RUN(X,Y,N)
      REAL X(N),Y(N)
      DO I=1,N
         X(I)=Y(I)
      ENDDO
      END
