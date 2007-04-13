      SUBROUTINE LECOPT ()
  
C     Bug in: DO   WHILE (TEXTLU(:4) .NE. 'FIN ') ?

      CHARACTER * 133  TEXTLU, TXLIGN
      COMMON  /REDTEX/ TEXTLU, TXLIGN(99)
  
      NPASSES = -1

C      IGRILLE=0
      IGRILLE=0

      DO   WHILE (TEXTLU(:4) .NE. 'FIN ')
           IF (TEXTLU(:4).EQ.'TRAC') THEN
                CALL REDLEV ('IC')
                DO WHILE (ITYPLU.EQ.1)
                     KFLU (INTEGR) = 1
                     CALL REDLEV ('IC')
                ENDDO
           ELSE IF (TEXTLU(:7).EQ.'CAMARET') THEN
                CALL REDLEV ('C')
           ELSE IF (TEXTLU(:4).EQ.'CTHE') THEN
                ITHERM  =  1
                CALL REDLEV ('C')
           ELSE IF (TEXTLU(:4).EQ.'BOMB') THEN
                NBOM  =  1
                CALL REDLEV ('C')
           ELSE IF (TEXTLU(:4).EQ.'FORC') THEN
                CALL REDLEV ('C')
           ELSE IF (TEXTLU(:4).EQ.'CAL1') THEN
                ICAL13 =  1
                CALL REDLEV ('C')
           ELSE IF (TEXTLU(:4).EQ.'RECA') THEN
                IRECA  =  1
              
           ELSE IF (TEXTLU(:4).EQ.'ETAL') THEN
                NETA = 1
                 ENDIF
      ENDDO
      END

