C     Simplified version of AXIAL, used to expose a bug in
C     transformers_inter_full, most likely linked to bourdoncle
C     restructuration.

                      SUBROUTINE  UNSTRUC19
     * ( DIRR, NUMIRR , NUMMAT , IBUR)
      LOGICAL FINDMO
      Real XIR (1)

    2 FINDMO=.FALSE.

    4 IF (FINDMO) THEN
         IR=IRINF
    5    CONTINUE
         IF (ITYP.EQ.2 .AND. IR.LE.IRSUP) THEN
            XIR(IR)=AFLOT
            IR=IR+1
            GOTO 5
         ENDIF

         GOTO 2
      ENDIF

      END
