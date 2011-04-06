C Looking for an example. Taken from apsi.f

C     FI: copied from costi.f, Modified back to the apsi version I believe

      SUBROUTINE COSTI02 (N,WSAVE)
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
      FK = 0.D0
      KC = NP1-1
      DO 101 K=2,NS2
         KC = NP1-K
         FK = FK+1.
         WSAVE(K) = 2.D0*DSIN(FK*DT)
         WSAVE(KC) = 2.D0*DCOS(FK*DT)
  101 CONTINUE
C      CALL RFFTI (NM1,WSAVE(N+1))
      RETURN
      END
