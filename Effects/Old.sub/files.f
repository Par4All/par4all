C     There should no parallelization possible
C     because of the write instruction ...
      PROGRAM FILES
      INTEGER I(10)
      INTEGER J

      DO 10 J=1,20
c         WRITE (UNIT=5,*) I(J)
         WRITE (UNIT=5,FMT=*) I(J)
 10   CONTINUE

      END
