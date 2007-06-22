C     This sub has the same structure as FFT
C     this program "core dumped"
C     During the semantical analysis, MMAX is unknown in the both loops 
C
C     Modifications:
C      - add a second IF THEN ENDIF to preserve the information about N and NN
      SUBROUTINE FSTR2(DATA,NN,ISIGN)
      REAL*8 WR,WI,WPR,WPI,WTEMP,THETA
      DIMENSION DATA(2*NN)
      N=2*NN
      MMAX=2
      IF(N.GT.MMAX) THEN
 2       IF(N.GT.MMAX) THEN
            ISTEP=2*MMAX
            WI=0.D0
            DO 13 M=1,MMAX,2
               DO 12 I=M,N,2*MMAX
                  J=I+MMAX
 12            CONTINUE
               WTEMP=WR
 13         CONTINUE
            MMAX=ISTEP
            GOTO 2
         ENDIF
         I=2
      ENDIF
      END
