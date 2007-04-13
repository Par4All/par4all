      PROGRAM NGA09

C     Bug: the inner loop may not be executed if rmin is greater than
C     rmax, which the compiler cannot guess. The transformer for the
C     loop nest cannot carry information about the value of t on exit,
C     especially since the lower loop bound is not analyzable.

C     Bug: in fact, the initial bug is linked to the unknown lower bound
C     already, regardless of the enclosing loop. It's easier to debug
C     with one fewer loops.
  
      REAL WLOOP(13,13)
      INTEGER R,T,RMIN,RMAX,TMIN(13),TMAX(13),WHICH(13,13)
 
      REAL FC(2)

c      DO 4 R=RMIN,RMAX
         DO 4 T=TMIN(R+1),TMAX(R+1)
            FC(2) = WLOOP(R+1,T+1)
            
 4    CONTINUE
C     
      END
      
