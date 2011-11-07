C Nga Nguyen: This example shows a bug in regions
C J is modified by the loop but the region does not take into account
C because J is not analysed by semantics analyses. 
C  
C    
      PROGRAM BUGREGION
      DIMENSION ITAB(2000)
      EQUIVALENCE (ITAB(10),J)
      J = 0
      DO I = 1,12
         J = J+1 
         ITAB(J) = 1000
      ENDDO
      END


