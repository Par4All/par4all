      PROGRAM IMPLIC
C     BUG: IMPLICIT DO IN IO LIST ARE NO LONGER ACCEPTED BY THE PARSER
      DIMENSION TIT(2,7)

      PRINT *, K,(TIT(L,N),L=1,2)
      WRITE(IMP,2000) K,(TIT(L,N),L=1,2)
 2000 FORMAT(1H1,44X,'   K=',I2,2A10,/
     1          ,45X,'------------------------------',/)

      END
