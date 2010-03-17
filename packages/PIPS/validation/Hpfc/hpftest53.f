c Massive I/O test taken from onde 24
      PROGRAM hpftest53
      INTEGER   NP
      PARAMETER (NP = 800)
      REAL*8    V     (NP,NP)
CHPF$ TEMPLATE DOMAIN(NP,NP)
CHPF$ PROCESSORS PE(4,4)
CHPF$ DISTRIBUTE DOMAIN(BLOCK,BLOCK) ONTO PE
CHPF$ ALIGN V(I,J) WITH DOMAIN(I,J)
      DO I = 1,NP
         DO J = 1,NP
            READ (3,*) V(I,J)
         ENDDO
      ENDDO
      END
