      PROGRAM LOOP02

C     Bug on transformer: you cannot say anything about T's value after
C     the loops. The loops seem to be supposed always entered and T's
C     value to always be increased because the transformer syntactically
C     associated to the loop is semantically associated to the loop
C     body.
  
      REAL WLOOP(13,13)
      INTEGER R,T,RMIN,RMAX,TMIN(13),TMAX(13),WHICH(13,13)
 
      REAL FC(2)

      DO 4 R=RMIN,RMAX
         DO 4 T=TMIN(R+1),TMAX(R+1)
            FC(2) = WLOOP(R+1,T+1)
            
 4    CONTINUE

      DO T = 1, N
         FC(I) = WLOOP(I,I)
      ENDDO

      DO T = 1, 2
         FC(I) = WLOOP(I,I)
      ENDDO
C     
      END
      
