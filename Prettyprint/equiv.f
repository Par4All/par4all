C     ensemble de routines pour tester le prettyprint des equivalences.

C     EQUIV1 provient de OCEAN
C     le tri des classes d'equivalences par offset croissant
C     permet de preserver les declarations du programme.

      subroutine EQUIV1

      PARAMETER (NX=128,NY=128)
      PARAMETER (NA1=2* (NY+1),NA2=NX/2,NE=NY-1,NW1=NY+1)
      PARAMETER (NW2=128,NEX=256)
      PARAMETER (NI=2*NY-1,NEQ=3*NY-2,NCW2=2*NW1+1,NCW3=4*NW1+1)

C     A-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
C     CA+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
C     EVAL+-+IR-+-+-IC+-+-+E+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
C
C
C     WORK+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
C     CWORK-+-+-+CWORK2-+-+-CWORK3+-+-+
C

      REAL A(NA1,NA2),E(NE,NE)
      REAL WORK(NW1*NW2)
      REAL EVAL(NE),IR(NE),IC(NE)
      COMPLEX CA(NW1,NA2),CWORK(NW1),CWORK2(NW1),CWORK3(NW1)
C
      EQUIVALENCE (A,CA,EVAL), (A(NY,1),IR), (A(NI,1),IC),
     +            (A(3*NE+1-NA1,2),E)
      EQUIVALENCE (WORK,CWORK), (WORK(NCW2),CWORK2), (WORK(NCW3),CWORK3)
      
      end

C     EQUIV2 est imaginaire, et permet de tester deux cas de figure.
C     Dans la premiere classe d'equivalence, le tableau B peut servir de 
C     base. Dans la deuxieme, c'est tour a tour D puis E.

      subroutine EQUIV2
      
      REAL A(5), B(10), C(10)
      REAL D(5), E(10), F(5)

      EQUIVALENCE (A,B), (B(6),C)
      EQUIVALENCE (D(2),E), (E(8),F)
      end

