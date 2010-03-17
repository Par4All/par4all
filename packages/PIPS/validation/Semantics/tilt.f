      SUBROUTINE  TILT(  )
       INTEGER IM1, JM1, KM1
      DO 100 K=1,KM1
         DO 200 J=1,JM1
            DO 200 I=1,IM1
              IF (Z.NE.0) GOTO 300
              P=1
              GO TO 200
 300          IF (Z2.NE.0) THEN
                 P=2
                 GO TO 200
              ENDIF
              RETURN
C     
 200       CONTINUE
 100    CONTINUE
C

c        PRINT *, P

      END
