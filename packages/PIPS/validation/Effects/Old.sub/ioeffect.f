C     Bug with IN regions, there is no IN region for NDIM

      PROGRAM IOEFFECT

      COMMON/PAR1/NDIM,LSIZE(4)

      READ(5,*)NDIM,(LSIZE(N),N=1,NDIM)

      END
