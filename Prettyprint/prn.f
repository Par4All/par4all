C                                                                       
C  *********************************************************************
C  *                  ROUTINE TO PRINT THE MEAN PROFILES               *
C  *********************************************************************
C                                                                       
      SUBROUTINE PRN(NZ,UM,VM,TM,DKM,ZMH,DL,USTAR,KLAS,WM,              
     *ZET,DNORM,ITR)                                                    
      REAL UM(1),VM(1),TM(1),ZET(1),DKM(1)                              
      COMPLEX WM(1)                                                     
      COMMON/LAKE/SX,SIGX,SY,SIGY,ZNOT,SFX,SFY,FLX,FLY,BASE,TSL,H0,ZW,ZL
CX30123 CALL SBINX (123)                                                
CY30123 CALL SBINY (123)                                                
      WRITE(6,10) ITR,DNORM,ZNOT,ZMH,USTAR,KLAS,DL                      
      WRITE(8,10) ITR,DNORM,ZNOT,ZMH,USTAR,KLAS,DL                      
      WRITE(8,20)                                                       
      WRITE(6,20)                                                       
 10   FORMAT(/,'  CONVERGENCE AFTER ',I6,' ITERATIONS.    NORM= ',E12.4,
     */,' Z0=',F8.4,'   ZMIX=',F7.2,'   U*=',F9.5,'   L(',I1,')=',F12.2)
 20   FORMAT(//,'   K ',1X,'  HEIGHT (M)',6X,' UM(Z) ',3X,' VM(Z) '     
     *,4X,' TM(Z) ',4X,'  KM(Z)  ',/)                                   
CX10663 CALL DOINX( 663 )                                               
CY10663 CALL DOINY( 663 )                                               
      DO 30 K=1,NZ                                                      
         WRITE(6,40) K,ZET(K),WM(K),TM(K),DKM(K)                        
 30   CONTINUE                                                          
CX20664 CALL DOOUTX( 664 )                                              
CY20664 CALL DOOUTY( 664 )                                              
 40   FORMAT(I4,F11.2,4X,2F10.2,F12.2,F12.2)                            
CX40124 CALL SBOUTX (124)                                               
CY40124 CALL SBOUTY (124)                                               
      RETURN                                                            
      END                                                               
