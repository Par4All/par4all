      SUBROUTINE VERIFY()
C                                                                       
C
C     VERIFY THE CORRECTNESS OF COMPUTATION.
C
C     ON ENTRY
C
C     EU1, EV1, EW1, EOX1, EOY1, EOZ1
C        ARE THE ARRAYS OF ENERGY.
C        REAL*8 EU1(12),EV1(12),EW1(12),EOX1(12),EOY1(12),EOZ1(12).
C
      IMPLICIT REAL*8(E-F)
C

      DIMENSION EU1(12),EV1(12),EW1(12),EOX1(12),EOY1(12),EOZ1(12)
      COMMON /ENG/ EU1, EV1, EW1, EOX1, EOY1, EOZ1
C
      COMMON /PAR1/ ISTART,NSTEPS,NAVG,ISAV,NSAV,NOUT,IRND,ITG,ITEST
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ
C
      OPEN(10, FILE='TURB3D.VERIFY')  
C.... THE VERIFICATION ROUTINE                                      
      WRITE(10,1)                                                       
    1 FORMAT(1X,'TURB3D BENCHMARK VERIFICATION & TIMING'/)              
      WRITE(10,2)                                                       
    2 FORMAT(1X,'VALIDATION PARAMETERS:'/)                              
      DO 20 I=1,11
         WRITE(10,3) EU1(I),EV1(I),EW1(I)
         WRITE(10,3) EOX1(I),EOY1(I),EOZ1(I)
 20   CONTINUE
    3 FORMAT(1X,3E20.12)                                                
C                                                                       
      IVALID=0                                                          
C
      IF((ABS(EU1(1)-0.625000000000E-01)/0.625000000000E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(1)-0.625000000000E-01)/0.625000000000E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(1))).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(1)-0.625000000000E-01)/0.625000000000E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(1)-0.625000000000E-01)/0.625000000000E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(1)-0.250000000000E+00)/0.250000000000E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(2)-0.624242648740E-01)/0.624242648740E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(2)-0.624242648740E-01)/0.624242648740E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(2)-0.155808881559E-05)/0.155808881559E-05).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(2)-0.624359507677E-01)/0.624359507677E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(2)-0.624359507677E-01)/0.624359507677E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(2)-0.249693943792E+00)/0.249693943792E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(3)-0.623470643533E-01)/0.623470643533E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(3)-0.623470643533E-01)/0.623470643533E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(3)-0.621437580180E-05)/0.621437580180E-05).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(3)-0.623936757947E-01)/0.623936757947E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(3)-0.623936757947E-01)/0.623936757947E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(3)-0.249375836204E+00)/0.249375836204E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(4)-0.622684059836E-01)/0.622684059836E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(4)-0.622684059836E-01)/0.622684059836E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(4)-0.139411938815E-04)/0.139411938815E-04).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(4)-0.623729831887E-01)/0.623729831887E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(4)-0.623729831887E-01)/0.623729831887E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(4)-0.249045779559E+00)/0.249045779559E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(5)-0.621882978354E-01)/0.621882978354E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(5)-0.621882978354E-01)/0.621882978354E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(5)-0.247099560140E-04)/0.247099560140E-04).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(5)-0.623736799074E-01)/0.623736799074E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(5)-0.623736799074E-01)/0.623736799074E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(5)-0.248703891007E+00)/0.248703891007E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(6)-0.621067484882E-01)/0.621067484882E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(6)-0.621067484882E-01)/0.621067484882E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(6)-0.384911706662E-04)/0.384911706662E-04).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(6)-0.623955717372E-01)/0.623955717372E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(6)-0.623955717372E-01)/0.623955717372E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(6)-0.248350302140E+00)/0.248350302140E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(7)-0.620237670172E-01)/0.620237670172E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(7)-0.620237670172E-01)/0.620237670172E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(7)-0.552544589510E-04)/0.552544589510E-04).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(7)-0.624384632917E-01)/0.624384632917E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(7)-0.624384632917E-01)/0.624384632917E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(7)-0.247985158598E+00)/0.247985158598E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(8)-0.619393629806E-01)/0.619393629806E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(8)-0.619393629806E-01)/0.619393629806E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(8)-0.749685727905E-04)/0.749685727905E-04).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(8)-0.625021580253E-01)/0.625021580253E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(8)-0.625021580253E-01)/0.625021580253E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(8)-0.247608619650E+00)/0.247608619650E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(9)-0.618535464078E-01)/0.618535464078E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(9)-0.618535464078E-01)/0.618535464078E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(9)-0.976014138093E-04)/0.976014138093E-04).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(9)-0.625864582582E-01)/0.625864582582E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(9)-0.625864582582E-01)/0.625864582582E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(9)-0.247220857761E+00)/0.247220857761E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(10)-0.617663277872E-01)/0.617663277872E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(10)-0.617663277872E-01)/0.617663277872E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(10)-0.123120052813E-03)/0.123120052813E-03).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(10)-0.626911652110E-01)/0.626911652110E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(10)-0.626911652110E-01)/0.626911652110E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(10)-0.246822058147E+00)/0.246822058147E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(11)-0.616777180541E-01)/0.616777180541E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(11)-0.616777180541E-01)/0.616777180541E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(11)-0.151490749774E-03)/0.151490749774E-03).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(11)-0.628160790475E-01)/0.628160790475E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(11)-0.628160790475E-01)/0.628160790475E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(11)-0.246412418305E+00)/0.246412418305E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF(IVALID.EQ.0) THEN                                              
         WRITE(10,4)                                                    
    4    FORMAT(//1X,'RESULTS FOR THIS RUN ARE:  VALID')                   
      ELSE                                                           
         WRITE(10,5)                                                    
    5    FORMAT(//1X,'RESULTS FOR THIS RUN ARE:  INVALID')                  
      ENDIF                                                             
C                                                                       
C                                                                       
      RETURN                                                              
C                                                                       
 4000 FORMAT (A,4x,I3,6x,2(1x,E12.4,' +/- ',E10.4))
C                                                                       
      END
