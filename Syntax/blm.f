C     Bug: if OVL is parsed first, its explicit declaration does not fit 
C     its implicit declaration in BLM

C     Extract from apsi (Spec CFP95)

      SUBROUTINE BLM(U,POTT,ZET,NZ,Z0,USTAR,DL,DZ,DKZ,KLAS,ZMH,F)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      REAL*8 POTT(*),ZET(*),DKZ(*)
            DL=1.D0/OVL(BETA,Z0)
      END
C
      FUNCTION OVL(BETA,Z0)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      OVL=MIN(XL,-0.003D0)
      END
