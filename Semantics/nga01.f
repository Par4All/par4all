      PROGRAM NGA01
C
  
      REAL WLOOP(13,13)
      INTEGER R,T,RMIN,RMAX,TMIN(13),TMAX(13),WHICH(13,13)
 
      REAL FC(2)
    
      DO 2 R=1,13
         DO 2 T=1,13
            WHICH(R,T) = 0
 2    CONTINUE

      DO 3 R=RMIN,RMAX
         DO 3 T=1,13
            IF((T.GE.TMIN(R+1)).AND.(T.LE.TMAX(R+1))) THEN
               WHICH(R+1,T+1) = 1
            ENDIF
 3    CONTINUE

      DO 4 R=RMIN,RMAX
         DO 4 T=TMIN(R+1),TMAX(R+1)
            FC(2) = WLOOP(R+1,T+1)
            
 4    CONTINUE
C     
      END
      
