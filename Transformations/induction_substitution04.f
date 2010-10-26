C Looking for an example. Taken from apsi.f

      SUBROUTINE COSTI_SIMPLIFIED (N,WSAVE)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DIMENSION       WSAVE(*)
      PARAMETER (PI = 3.14159265358979)
      NP1 = N+1
      NS2 = N/2
      DT = PI/FLOAT(NM1)
      KC = NP1-1
      DO 101 K=2,NS2
         KC = KC - 1
         WSAVE(KC) = 2.D0*DCOS(DT)
  101 CONTINUE
      RETURN
      END
