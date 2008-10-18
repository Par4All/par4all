* Example with sequence and unstructured

      PROGRAM SEQUENCE
      REAL X(500), Y(500)
      READ *,A,B
      N = 0
 100  IF (A.GT.0.0) THEN
         N=N+1
         X(N)=A
         Y(N)=B
         READ *,A,B
         GOTO 100
      ENDIF
      S=0.0
      DO 200 I=1,N
         S=S+X(I)*Y(N)
 200  CONTINUE
      WRITE (*,'("b",F15.5)')S
      END



