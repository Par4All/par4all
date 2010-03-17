      SUBROUTINE  TYPE01
     &    (FIJK,ALPX,SOUF,SOUX,ALPY,ALPZ,DIAG,GRAD,DIAGS,ADIR,ZZ,DIAGIN,
     &                                                             TETA)

C     Bug: the type for SDOT is wrong, the parameter list is assumed empty

      INTEGER NMOT
      REAL GRAD(1),ZZ(1)
      REAL    SDOT
      XMUN=SDOT(NMOT,GRAD,1,ZZ,1)

      END
