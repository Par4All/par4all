       SUBROUTINE  TYPE02
     &    (FIJK,ALPX,SOUF,SOUX,ALPY,ALPZ,DIAG,GRAD,DIAGS,ADIR,ZZ,DIAGIN,
     &                                                             TETA)
      INTEGER NMOT
      REAL GRAD(1),ZZ(1)
      XMUN=SDOT(NMOT,GRAD,1,ZZ,1)

      END
