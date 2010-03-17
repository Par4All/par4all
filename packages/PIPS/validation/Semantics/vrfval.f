      SUBROUTINE VRFVAL ()
 
C     Bug found by Corinne on 4 October 2002 (not reproduced)

      REAL*8 X,Y,I      

      IF (MAX(ABS(X),ABS(Y)).GT.1.) I = 1

      Z = ABS(X)
      U = ABS(Y)
      V = AMAX1(Z, U)

      if(V.gt.1.) then
C     Let's see what information is generated:
         print *, x, y, z, u, v
C     And what's left about X and Y? Nothing!
         read *, z, u, v
         print *, x, y
      else
C     Let's see again what information is generated:
         print *, x, y, z, u, v
C     And what's left about X and Y? Something!
         read *, z, u, v
         print *, x, y
      endif

      print *, x, y, z, u, v
      END
