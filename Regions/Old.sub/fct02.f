C     Check on block transformer, same as fct.f, but no wasted statement
C     such as return or continue.
C
C     The problem: no information about NQ and MQ or their relationship
C
C     The information gathered about the first loop is killed by the
C     third loop nest which can be entered or not. If it is not entered
C     I is preserved. If it is entered, I>=1 and I>=MQ+1. But there is
C     no way to know this is exactly the information gathered
C     before. Unless you use transformers computed in context.

      SUBROUTINE FCT02(UTRA)
      IMPLICIT REAL*8 (A-H, O-Z)
      PARAMETER  (MP = 402, NP = 160)
      COMMON /ADVC/  MQ,NQ
      DIMENSION UTRA(MP,NP)
      DIMENSION B(NP),A(MP)

      DO I=1,MQ
         A(I)=1.0D0
      ENDDO

      DO J=1,NQ
         B(J)=1.0D0
      ENDDO

      DO J = 1,NQ
         DO I = 1,MQ
            UTRA(I,J) =  B(J) + A(I)
         ENDDO
      ENDDO

      END
