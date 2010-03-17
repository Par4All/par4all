      PROGRAM MAIN
      CALL ADVNCE
      END
      
      SUBROUTINE ADVNCE
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      COMMON /VAR1/   RO (MP,NP)
      COMMON /VAR2/   RON(MP,NP)
      CALL FCT(RON, RON, RO)
      END

      SUBROUTINE FCT ( UNEW, UTRA, UOLD )
      PARAMETER  ( MP   =      402 ,    NP   =       160 )
      DIMENSION  UNEW(MP,NP), UTRA(MP,NP), UOLD(MP,NP)
      COMMON /ADVC/   MQ, NQ
      DO J = 1, NQ
         DO I = 1, MQ
            UTRA(I,J) = UTRA(I,J)+ 1
200         CONTINUE
         ENDDO
      ENDDO
      DO J = 1, NQ
         DO I = 1, MQ
            UNEW(I,J) = UTRA(I,J)- 1
400         CONTINUE
         ENDDO
      ENDDO
      END
