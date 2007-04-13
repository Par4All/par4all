      PROGRAM UNSTRUC18

C     Bug in bourdoncle.c, encountered by Nga

      REAL X(513,513)
      INTEGER I,J,K,N
      READ *,N
      DO I = 1,N
         DO J = 1,N
            READ (*,ERR=990) X(I,J)
         ENDDO
      ENDDO
      GOTO 2
 990  STOP 'PROBLEM WITH INPUT FILE: STOP'
      PRINT *,'OKAY: CONTINUE'
 2    END
