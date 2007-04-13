C
C **********************************************************************
C *            CONVERSION TO MEAN U,V COMPONENTS                       *
C **********************************************************************
C
      SUBROUTINE FINAL(NZ,UM,VM,UG,VG,TM,DKM,ZMH,DL,USTAR,KLAS,WM,
     *ZET,DNORM,ITR,NY,DT,TIMEM)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      DIMENSION UM(1),VM(1),TM(1),ZET(1),DKM(1),UG(1),VG(1)
      COMPLEX*16 WM(1)
      COMMON/LAKE/SX,SIGX,SY,SIGY,ZNOT,SFX,SFY,FLX,FLY,BASE,TSL,H0,ZW,ZL
CX30113 CALL SBINX (113)
CY30113 CALL SBINY (113)
      WRITE(6,10) ITR,DNORM,TIMEM,ZNOT,ZMH,USTAR,KLAS,DL
      WRITE(8,10) ITR,DNORM,TIMEM,ZNOT,ZMH,USTAR,KLAS,DL
      WRITE(8,20)
      WRITE(6,20)
 10   FORMAT(/,
     *' ***********************  MEAN PROFILES ***********************'
     *,//,' CONVERGENCE AFTER ',I6,' ITERATIONS.    NORM= ',E12.4,
     */,' EQUIVALENT TIME FOR THE MEAN PROFILES  IS ',F10.2,' SEC. ',
     */,' Z0=',F8.4,'   ZMIX=',F7.2,'   U*=',F9.5,'   L(',I1,')=',F12.2)
 20   FORMAT(//,'   K ',1X,'  HEIGHT (M)',6X,' UG(Z) ',3X,' VG(Z) '
     *,4X,' TM(Z) ',4X,'  KM(Z)  ',4X,' UM(Z)  ',4X,' VM(Z)  ',/)
CX10649 CALL DOINX( 649 )
CY10649 CALL DOINY( 649 )
      DO 30 K=1,NZ
         UM(K)=DBLE(WM(K))
         VM(K)=DIMAG(WM(K))
         WRITE(6,40) K,ZET(K),UG(K),VG(K),TM(K),DKM(K),UM(K),VM(K)
         WRITE(8,40) K,ZET(K),UG(K),VG(K),TM(K),DKM(K),UM(K),VM(K)
 30   CONTINUE
CX20650 CALL DOOUTX( 650 )
CY20650 CALL DOOUTY( 650 )
 40   FORMAT(I4,F11.2,4X,2F10.2,4F12.2)
      WRITE(6,50)
      WRITE(8,50)
 50   FORMAT(/,
     *' **************************************************************'
     *,/)
CX40114 CALL SBOUTX (114)
CY40114 CALL SBOUTY (114)
      RETURN
      END
