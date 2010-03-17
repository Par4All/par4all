C
C DOUBLE COMPLEX EXTENSION
C
      PROGRAM P
      DOUBLE COMPLEX DC1, DC2(3)
      COMPLEX*16 DC3
      INTEGER I
      DC1 = (1.0D0,1.0D0)
      DC3 = 0
      DO 10 I=1, 3
         DC2(I) = DC1 + I
         DC3 = DC3 + DC2(I)
 10   CONTINUE
      PRINT *, DC3
      END
