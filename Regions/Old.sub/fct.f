C     Nga Nguyen 
C     This example is extracted from hydro2d.f (SPEC CFP95)
C     False result: IN EXACT for B at module statement
C     Reason : preconditions have loop exit conditions but transformers do not. 

      SUBROUTINE FCT (UTRA)
      IMPLICIT REAL*8 (A-H, O-Z)
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      COMMON /ADVC/  MQ,NQ
      DIMENSION UTRA(MP,NP)
      DIMENSION B(NP),A(MP)

      DO 20 I=1,MQ
         A(I)=1.0D0
   20 CONTINUE

      DO 40 J=1,NQ
         B(J)=1.0D0
   40 CONTINUE

      DO 200  J = 1,NQ
      DO 200  I = 1,MQ
         UTRA(I,J) =  B(J) + A(I)
  200 CONTINUE

      RETURN
      END
