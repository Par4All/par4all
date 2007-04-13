C     Bug: one of the two invariants is lost

C     Modification: the loop body is shortened
      
C     MODIFIED VERSION CONVGF: THE TWO INTERNAL LOOPS ARE FUSIONED
      
      SUBROUTINE CONVGFM(WKE, WKS, LST, NX, KRN, SCALE, KX, KY)
      REAL WKE(*)
      REAL WKS(*)
      REAL KRN(*)
      INTEGER VOIS, OFFTAB, COEFDEB, COEF
      
      IKEND = LST                                  
      OFFTAB = 2*NX/2                              
      COEFDEB = 3*IKEND
      IWKE = 0         
      IWKS = 0         
      DO I = 0, NX-1   
         IWKE = IWKE+1
         IWKS = IWKS+1        
      ENDDO
      END
