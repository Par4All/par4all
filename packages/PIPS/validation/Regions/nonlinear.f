      PROGRAM NONLINEAR
      DIMENSION ITAB(2000)
      DO I = 1,12
         ITAB(I/2) = 1000
C         ITAB(I/3 +1) = 1000
C         ITAB(MOD(I,3)) = 1000
      ENDDO
      END
