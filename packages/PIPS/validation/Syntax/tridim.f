      program tridim
      PARAMETER      (NPMAX     = 809)
      REAL*8    U     (NPMAX+1,NPMAX,2)

      DO 100 i = 1, 10
         U(i,2,3) = 0.
 100  continue
      end
