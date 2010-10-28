C Looking for an example. Taken from apsi.f

      SUBROUTINE COSTI (N,WSAVE)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DIMENSION       WSAVE(*)
      PARAMETER (PI = 3.14159265358979)
      IF (N .LE. 3) THEN
      RETURN
      ENDIF
      NM1 = N-1
      NP1 = N+1
      NS2 = N/2
      DT = PI/FLOAT(NM1)
c      FK = 0.D0
      KC = NP1-1
      DO 101 K=2,NS2
C         KC = NP1-K
         KC = KC - 1
C         FK = FK+1.
         FK = FLOAT(K-1)
         WSAVE(K) = 2.D0*DSIN(FK*DT)
         WSAVE(KC) = 2.D0*DCOS(FK*DT)
  101 CONTINUE
C      CALL RFFTI (NM1,WSAVE(N+1))
      RETURN
      END
