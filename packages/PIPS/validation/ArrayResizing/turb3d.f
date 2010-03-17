C******************************************************************************C 
C                                                                              C
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++C
C                                                                              C
C CODE NAME : TURB3D                                                           C
C                                                                              C
C PURPOSE   :                                                                  C
C            This program can be used for simulating isotropic,                C
C            homogeneous turbulence in a cube with periodic                    C
C            boundary conditions in x,y,z coordinate directions.               C
C            It solves the Navier-Stokes equations using a pseudo-             C
C            spectral method. Leapfrog-Crank-Nicolson scheme is                C
C            used for time stepping.                                           C
C                                                                              C
C AUTHOR    :                                                                  C
C            TURB3D was written by Dr. Raj Panda                               C
C                                                                              C
C HISTORY   :                                                                  C
C            The original version was written for IBM 3090 systems in 1989     C
C            and utilized the FFT routines from IBM's ESSL. It was modified by C	
C            David Schneider and Hui Gao at CSRD to run with David Bailey's    C 
C            FFT routines. This version is the CSRD version with minor changes.C
C                                                                              C 
C            At CSRD, TURB3D was run on different platforms including,         C
C            IBM RS/6000, Sun-4, Alliant FX/80 & FX/2800 and Cray Y-MP.        C 
C                                                                              C
C SIZE      :                                                                  C
C            All benchmarking should be done at size 64*64*64                  C
C                                                                              C
C PARALLEL  :                                                                  C
C            The parallel content in the code is close to 100%.                C 
C +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++C
C                                                                              C
C                                                                              C 
C******************************************************************************C 

C                                                                       
C
C     SUBROUTINE CALLS: PBRINT, TURB3D, PARAM, TGVEL, 
C                        WCAL, DCOPY, ENR, ZFFT, XYFFT, UXW, LIN, 
C                        LINAVG, MIXAVG, VERIFY.
C
C


      PROGRAM MAIN
      COMMON /PAR1/ ISTART,NSTEPS,NAVG,ISAV,NSAV,NOUT,IRND,ITG,ITEST 
C SPEC, JWR: Line copied from line 75 to be sure ITEST is defined
C
      CALL TURB3D()
      IF(ITEST.EQ.1) THEN
      CALL VERIFYTR()
      ELSE
      CALL VERIFY()
      ENDIF
      STOP
      END 
C
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE TURB3D() 
C***                                                                    
      IMPLICIT REAL*8(A-H,O-Z)                                          
      PARAMETER   (IX= 64,IY= 64,IZ= 64 ,M1=6)

      PARAMETER   (IXPP=IX+2,IXHP=IX/2+1,ITOT=IXPP*IY*IZ,               
     +          IXY=IXHP*IY,IXZ=IXHP*IZ,IYZ=IY*IZ)                      
      COMMON/ALL/ U(IXPP,IY,IZ),V(IXPP,IY,IZ),W(IXPP,IY,IZ),            
     +            OX(IXPP,IY,IZ),OY(IXPP,IY,IZ),OZ(IXPP,IY,IZ)          
      COMMON/SHR/ U0(IXPP,IY,IZ),V0(IXPP,IY,IZ),W0(IXPP,IY,IZ),         
     +            U1(IXPP,IY,IZ),V1(IXPP,IY,IZ),W1(IXPP,IY,IZ)          
C
      COMMON/WAV/ XW(129),YW(129),ZW(129),XSQ(129),YSQ(129),ZSQ(129)    
      COMMON/TRUNC/XWT(129),YWT(129),ZWT(129)                           
      COMMON /BOX/ PI,PI2,BOXX,BOXY,BOXZ,FX,FY,FZ                       
      COMMON /PAR1/ ISTART,NSTEPS,NAVG,ISAV,NSAV,NOUT,IRND,ITG,ITEST          
      COMMON /PAR2/ REY,XNU,DT                                          
      COMMON /COUNT/ KTTRANS(256)
      COMMON /DIM/ ISIZE(9)                                             
C                                                                       
      DIMENSION EU1(12),EV1(12),EW1(12),EOX1(12),EOY1(12),EOZ1(12)
      COMMON /ENG/ EU1, EV1, EW1, EOX1, EOY1, EOZ1
C
C                                                                       
C      OPEN(5, FILE='TURB3D.INPUT')                                    
C      OPEN(6, FILE='TURB3D.OUTPUT')                                   
C
      WRITE(6,2010)                                                     
      WRITE(6,2000)                                                     
      WRITE(6,2010)                                                     
C
      NX   = IX                                                         
      NY   = IY                                                         
      NZ   = IZ                                                         
      NXPP = IXPP                                                       
      NXHP = IXHP                                                       
      NTOT = ITOT                                                       
      NXY  = IXY                                                        
      NXZ  = IXZ                                                        
      NYZ  = IYZ                                                        
C
      ISIZE(1) = NX                                                     
      ISIZE(2) = NY                                                     
      ISIZE(3) = NZ                                                     
      ISIZE(4) = NXPP                                                   
      ISIZE(5) = NXHP                                                   
      ISIZE(6) = NTOT                                                   
      ISIZE(7) = NXY                                                    
      ISIZE(8) = NXZ                                                    
      ISIZE(9) = NYZ                                                    
C
      DO 555 I=1,256
        KTTRANS(I) = 0
 555  CONTINUE
C
      WRITE (6,3000) (ISIZE(I),I=1,3)                                   
      ISTEP0 = 0
      CALL PARAM                                                        
      DT2 = DT                                                          
      IF(ISTART.EQ.1) DT2 = 2.D0*DT                                     
      IF(ISTART.EQ.0) THEN                                              
         JSTEP = 1                                                      
         ISTEP = 0                                                      
         NLAST = ISTEP + NSTEPS                                         
         TIME = 0.0D0                                                   
         IF(IRND.EQ.1) THEN                                             
            STOP '*** RANDOM VELOCITY FIELD NOT ALLOWED ***'                           
         ENDIF                                                          
         IF(ITG.EQ.1) THEN                                              
            CALL TGVEL(U,V,W)                                           
            CALL WCAL (U,V,W,OX,OY,OZ)                                  
         ENDIF                                                          
      ELSE                                                           
         NLAST = ISTEP0 + NSTEPS                                    
         JSTEP = 1                                                  
         ISTEP = ISTEP0                                             
C
C      FOR RESTART USER MUST INPUT U,V,W HERE                     
C                                                                 
         CALL WCAL(U,V,W,OX,OY,OZ)                                  
      ENDIF                                                             
C
      CALL DCOPY(NTOT,U,1,U0,1)                                         
      CALL DCOPY(NTOT,V,1,V0,1)                                         
      CALL DCOPY(NTOT,W,1,W0,1)                                         
C
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
C     ::::::::::BEGIN TIME INTEGRATION LOOP::::::::::::::::::::::::::::
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
C
 1001 CONTINUE                                                          
C                                                                       
C                                                                       
      DTV = XNU                                                         
      DTT = 0.5D0*DT2                                                   
C     ***************************************************************** 
      IF(MOD(ISTEP,NOUT).EQ.0.OR.ISTEP.EQ.0) THEN                       
          MSTEP = INT(ISTEP/NOUT)
         IF(ITG.EQ.1) THEN                                              
            CALL ENR (U,U,0.5D0,EU)                                     
            CALL ENR (V,V,0.5D0,EV)                                     
            CALL ENR (W,W,0.5D0,EW)                                     
            CALL ENR (OX,OX,0.5D0,EOX)                                  
            CALL ENR (OY,OY,0.5D0,EOY)                                  
            CALL ENR (OZ,OZ,0.5D0,EOZ)                                  
            EU1(MSTEP+1)=EU
            EV1(MSTEP+1)=EV
            EW1(MSTEP+1)=EW
            EOX1(MSTEP+1)=EOX
            EOY1(MSTEP+1)=EOY
            EOZ1(MSTEP+1)=EOZ
         ENDIF                                                          
      ENDIF                                                             
C
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
C     :::::: FOURIER-TO-PHYSICAL SPACE FFT :::::::::::::::::::::::::::: 
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
C
C
      DO 101 J=1,NY                                                     
         CALL ZFFT(J,1,1,1)                                            
         CALL ZFFT(J,1,2,1)                                            
         CALL ZFFT(J,1,3,1)                                            
         CALL ZFFT(J,1,4,1)                                            
         CALL ZFFT(J,1,5,1)                                            
         CALL ZFFT(J,1,6,1)                                            
  101 CONTINUE                                                          
C
C
      DO 201 K=1,NZ                                                     
         CALL XYFFT(K,1,1,1)                                           
         CALL XYFFT(K,1,2,1)                                           
         CALL XYFFT(K,1,3,1)                                           
         CALL XYFFT(K,1,4,1)                                           
         CALL XYFFT(K,1,5,1)                                           
         CALL XYFFT(K,1,6,1)                                           
  201 CONTINUE                                                          
C
C
C
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
C     ::::::: NONLINEAR TERM IN PHYSICAL SPACE :::::::::::::::::::::::::
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
C
C
      CALL UXW(U,V,W,OX,OY,OZ)                                          
C
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
C     ::::::: PHYSICAL-TO-FOURIER SPACE FFT ::::::::::::::::::::::::::::
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
C
C
      DO 301 K=1,NZ                                                     
         CALL XYFFT(K,1,1,-1)                                           
         CALL XYFFT(K,1,2,-1)                                           
         CALL XYFFT(K,1,3,-1)                                           
         CALL XYFFT(K,1,4,-1)                                           
         CALL XYFFT(K,1,5,-1)                                           
         CALL XYFFT(K,1,6,-1)                                           
  301 CONTINUE                                                          
C
C
      DO 401 J=1,NY                                                     
         CALL ZFFT(J,1,1,-1)                                            
         CALL ZFFT(J,1,2,-1)                                            
         CALL ZFFT(J,1,3,-1)                                            
         CALL ZFFT(J,1,4,-1)                                            
         CALL ZFFT(J,1,5,-1)                                            
         CALL ZFFT(J,1,6,-1)                                            
  401 CONTINUE                                                          
C
C
C
      IF (MOD(ISTEP,2).EQ.0) THEN                                       
         CALL DCOPY(NTOT,U,1,U1,1)                                      
         CALL DCOPY(NTOT,V,1,V1,1)                                      
         CALL DCOPY(NTOT,W,1,W1,1)                                      
         CALL DCOPY(NTOT,U0,1,U,1)                                      
         CALL DCOPY(NTOT,V0,1,V,1)                                      
         CALL DCOPY(NTOT,W0,1,W,1)                                      
         ELSE                                                           
         CALL DCOPY(NTOT,U,1,U0,1)                                      
         CALL DCOPY(NTOT,V,1,V0,1)                                      
         CALL DCOPY(NTOT,W,1,W0,1)                                      
         CALL DCOPY(NTOT,U1,1,U,1)                                      
         CALL DCOPY(NTOT,V1,1,V,1)                                      
         CALL DCOPY(NTOT,W1,1,W,1)                                      
      ENDIF                                                             
C
C
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
C     :::::::::::::: TIME-STEPPING :::::::::::::::::::::::::::::::::::: 
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
C                                                                       
C     STABILIZE LEAPFROG TIME-STEPPING SCHEME                           
C                                                                       
C
      IF (JSTEP.EQ.NAVG) THEN                                           
C
C
         CALL LINAVG(U,V,W,OX,OY,OZ,DT2,DTV,DTT)                  
         IF (MOD(ISTEP,2).EQ.0) THEN                                
               CALL DCOPY(NTOT,U1,1,U,1)                                  
               CALL DCOPY(NTOT,V1,1,V,1)                                  
               CALL DCOPY(NTOT,W1,1,W,1)                                  
         ELSE                                                     
               CALL DCOPY(NTOT,U0,1,U,1)                                  
               CALL DCOPY(NTOT,V0,1,V,1)                                  
               CALL DCOPY(NTOT,W0,1,W,1)                                  
         ENDIF                                                      
         CALL MIXAVG(U,V,W,OX,OY,OZ)                              
C
C
C
         JSTEP = JSTEP + 1                                        
         DT2   = 2.D0*DT                                          
C
C
         GOTO 1001                                                 
      ELSE                                                           
C
C
         CALL LIN (U,V,W,OX,OY,OZ,DT2,DTV,DTT)                          
C
C
C
         IF(MOD(ISTEP,NOUT).EQ.0.OR.ISTEP.EQ.0) THEN                       
            IF(ITG.EQ.1) THEN                                              
               WRITE (6,4000)ISTEP,TIME                                    
               WRITE (6,4100) EU,EV,EW                                     
               WRITE (6,4200) EOX,EOY,EOZ                                  
            ENDIF
         ENDIF
         TIME = TIME + DT                                               
         DT2 = 2.D0*DT                                                  
         ISTEP = ISTEP + 1                                              
         JSTEP = JSTEP + 1                                              
         IF(MOD(ISTEP,NAVG).EQ.0) THEN                                  
            JSTEP = 1                                                   
         ENDIF                                                          
      END IF                                                            
C
C
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  
C     :::::::: END OF TIME-STEPPING ::::::::::::::::::::::::::::::::::  
C     ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  
C
C
      IF (ISTEP.LT.NLAST) GO TO 1001
 1111 CONTINUE
      ISUM = 0
      DO 556 I=1,256
       ISUM = ISUM + KTTRANS(I)
 556  CONTINUE
      WRITE(6,5555) ISUM
C
 2000 FORMAT (49H ::SIMULATION OF ISOTROPIC, DECAYING TURBULENCE::)     
 2010 FORMAT (49H ::::::::::::::::::::::::::::::::::::::::::::::::)     
 3000 FORMAT (//,                                                       
     +'      NX      NY      NZ',                                       
     + /,3I8)                                                           
 4000 FORMAT (' ISTEP=',I4,2X,'TIME=',F8.4/)                             
 4100 FORMAT (' EU,EV,EW    =',3E20.12)
 4200 FORMAT (' EOX,EOY,EOZ =',3E20.12)
 5555 FORMAT (//' TOTAL NUMBER OF TRANSPOSES PERFORMED = ',I10)
C                                                                       
      RETURN                                                              
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE  PARAM                                                 
C                                                                       
C     SET UP A PERIODIC BOX OF VOLUME FX, FY,FZ                        
C     TO READ INPUT FILE                                                
C                                                                       
C     SET UP A PERIODIC BOX OF VOLUME BOXX, BOXY,BOXZ.
C     COMPUTE FX = 2*PI/BOXX,
C             FY = 2*PI/BOXY,
C             FZ = 2*PI/BOXZ.
C     READ INPUT FILE AND CALL WAVNUM ROUTINE TO SET UP 
C     WAVE NUMBERS.                                               
C
C     SUBROUTINE CALL: WAVNUM.
C  
C                                                                       
      IMPLICIT REAL*8(A-H,O-Z)                                          
C                                                                       
C     REY      = REYNOLDS NO.                                           
C     XNU       = VISCOSITY                                             
C     BOXX                                                              
C     BOXY     = DIMENSIONS OF BOX                                      
C     BOXX                                                              
C     DT       = TIME STEP                                              
C     NSTEPS   = MAXIMUM NUMBER OF TIME STEPS                           
C     NAVG     = LEAPFROG AVERAGING INTERVAL                            
C     NOUT     = POST PROCESSING SAVE INTERVAL(DATA SAVED IN FOUR SPACE)
      COMMON/WAV/ XW(129),YW(129),ZW(129),XSQ(129),YSQ(129),ZSQ(129)    
      COMMON /PAR1/ ISTART,NSTEPS,NAVG,ISAV,NSAV,NOUT,IRND,ITG,ITEST          
      COMMON /PAR2/ REY,XNU,DT                                          
      COMMON/ENCON/ C1,C2                                               
      COMMON /ALIAS/ IALIAS                                             
      COMMON /BOX/ PI,PI2,BOXX,BOXY,BOXZ,FX,FY,FZ                       
      CHARACTER*80 JNK                                                  
C                                                                       
      PI = 2.0D0*ASIN (1.0D0)                                           
      PI2 = 2.0D0*PI                                                    
      BOXX = PI2                                                        
      BOXY = PI2                                                        
      BOXZ = PI2                                                        
      FX = PI2/BOXX                                                     
      FY = PI2/BOXY                                                     
      FZ = PI2/BOXZ                                                     
C                                                                       
      READ(5,100)JNK                                                    
      READ(5,*)ISTART,NSTEPS,NAVG,NOUT,IALIAS,ITEST                           
      READ(5,100)JNK                                                    
      READ(5,*)XNU,DT                                                   
      READ(5,100)JNK                                                    
      READ(5,*)IRND,ITG                                                 
      READ(5,100)JNK                                                    
      READ(5,*)C1,C2                                                    
      REY = 1.D0/XNU                                                    
  100 FORMAT(A80)                                                       
      WRITE (6,2000) ISTART,NSTEPS,NAVG,NOUT,IALIAS,ITEST                     
      WRITE (6,3000) REY,XNU,DT                                         
      IF(IRND.EQ.1) THEN                                                
      WRITE (6,*)'ENERGY SPECTRUM E(K) OF THE RANDOM INITIAL VELOCITY'  
         WRITE(6,5000)C1,C2                                             
 5000 FORMAT(/,' E(K) = ',F6.4,'*K**4*EXP(-',F6.4,'*K**2)')             
      ENDIF                                                             
      IF(ITG.EQ.1) THEN                                                 
         WRITE (6,*)'INITIAL VELOCITY IS A TAYLOR-GREEN VORTEX'         
      ENDIF                                                             
      CALL WAVNUM                                                       
      RETURN                                                            
 2000 FORMAT (/,'    ISTART    ',                                       
     +      '   NSTEPS    NAVG      NOUT      IALIAS     ITEST',/,6I10)           
 3000 FORMAT (/,'    REY       XNU       DT   ',                        
     +        /,3F10.5/)                                                
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE  WAVNUM                                                
C                                                                       
C                                                                       
C     TO SET UP WAVE NUMBERS XW,YW,ZW;  
C     SQUARE OF WAVE NUMBERS XSQ,YSQ,ZSQ;
C     AND TRUNC NUMBERS XWT,YWT,ZWT. 
C                                                                       
      IMPLICIT REAL*8(A-H,O-Z)                                          
      COMMON/WAV/ XW(129),YW(129),ZW(129),XSQ(129),YSQ(129),ZSQ(129)    
      COMMON/TRUNC/XWT(129),YWT(129),ZWT(129)                           
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      COMMON /BOX/ PI,PI2,BOXX,BOXY,BOXZ,FX,FY,FZ                       
      COMMON /ALIAS/ IALIAS                                             
      REAL*8 LIMX,LIMY,LIMZ                                             
      DATA   TINY/1.D-20/                                               
      NXH = NX/2                                                        
      DO 1 I=1,NXH                                                      
         XW(I) = FX*FLOAT(I - 1) + TINY                                 
         XSQ(I) = XW(I)*XW(I) + TINY                                    
    1 CONTINUE                                                          
      XW(NXHP) = TINY                                                   
      XSQ(NXHP) = TINY                                                  
      NYH = NY/2                                                        
      DO 2 J=2,NYH                                                      
         YW(J) = FY*FLOAT(J - 1)+ TINY                                  
         YW(NY+2-J) = -YW(J)                                            
         YSQ(J) = YW(J)*YW(J)                                           
         YSQ(NY+2-J) = YSQ(J)                                           
    2 CONTINUE                                                          
      YW(NYH+1) = TINY                                                  
      YW(1) = TINY                                                      
      YSQ(1) = TINY                                                     
      YSQ(NYH+1) = TINY                                                 
      NZH = NZ/2                                                        
      DO 3 J=2,NZH                                                      
         ZW(J) = FZ*FLOAT(J - 1) + TINY                                 
         ZW(NZ+2-J) = -ZW(J)                                            
         ZSQ(J) = ZW(J)*ZW(J)                                           
         ZSQ(NZ+2-J) = ZSQ(J)                                           
    3 CONTINUE                                                          
      ZW(NZH+1) = TINY                                                  
      ZSQ(NZH+1) = TINY                                                 
      ZW(1) = TINY                                                      
      ZSQ(1) = TINY                                                     
      DO 4 I=1,NXHP                                                     
    4 XWT(I) = 1.0D0                                                    
      DO 5 J=1,NY                                                       
    5 YWT(J) = 1.0D0                                                    
      DO 6 K=1,NZ                                                       
    6 ZWT(K) = 1.0D0                                                    
      LIMX = 2.D0*FLOAT(NXHP)/3.D0                                      
      LIMY = 2.D0*FLOAT(NY/2+1)/3.D0                                    
      LIMZ = 2.D0*FLOAT(NZ/2+1)/3.D0                                    
      IF(IALIAS.EQ.1) THEN                                              
         DO 41 I=1,NXHP                                                 
            IF(ABS(XW(I)).GT.LIMX) THEN                                 
               XWT(I) = 0.D0                                            
            ENDIF                                                       
   41    CONTINUE                                                       
         DO 51 J=1,NY                                                   
            IF(ABS(YW(J)).GT.LIMY) THEN                                 
               YWT(J) = 0.D0                                            
            ENDIF                                                       
   51    CONTINUE                                                       
         DO 61 K=1,NZ                                                   
            IF(ABS(ZW(K)).GT.LIMZ) THEN                                 
               ZWT(K) = 0.D0                                            
            ENDIF                                                       
   61    CONTINUE                                                       
      ENDIF                                                             
      RETURN                                                            
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
C  INITIAL CONDITION FOR A TAYLOR-GREEN VORTEX                           
C                                                                       
      SUBROUTINE  TGVEL (U,V,W)                                         
C
C     TO SET UP INITIAL CONDITIONS FOR A TAYLOR-GREEN VORTEX                  
C     U(I,Y,Z) = COS X SIN Y COS Z                                      
C     V(X,Y,Z) = - SIN X COS Y COS Z                                    
C     W(X,Y,Z) = 0
C                                                                       
C     ON RETURN
C
C     U, V, W 
C        ARE THE THREE DIMENSIONAL ARRAYS. SPECIFIED AS: 
C        COMPLEX*16 U(NXHP,NY,NZ),V(NXHP,NY,NZ),W(NXHP,NY,NZ) 
C
      IMPLICIT REAL*8(A-H,O-Z)                                          
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      COMPLEX*16  U(NXHP,NY,*),V(NXHP,NY,*),W(NXHP,NY,*),IUNIT          
      COMMON /PAR1/ ISTART,NSTEPS,NAVG,ISAV,NSAV,NOUT,IRND,ITG,ITEST          
C     VELOCITY FIELD IS SET UP IN FOURIER SPACE                         
C     U(X,Y,Z) = COS X SIN Y COS Z                                      
C     V(X,Y,Z) = - SIN X COS Y COS Z                                    
C     W(X,Y,Z) = 0                                                      
      IUNIT =  CMPLX(0.0D0,1.0D0)                                       
      DO 1 I=1,NXHP                                                     
         DO 2 J=1,NY                                                    
            DO 3 K=1,NZ                                                 
               U(I,J,K) = CMPLX (0.0D0,0.0D0)                           
               V(I,J,K) = CMPLX (0.0D0,0.0D0)                           
               W(I,J,K) = CMPLX (0.0D0,0.0D0)                           
    3       CONTINUE                                                    
    2    CONTINUE                                                       
    1 CONTINUE                                                          
      SCALAR=0.125D0                                                    
      U(2,2,2)=-SCALAR*IUNIT                                            
      U(2,NY,2)=SCALAR*IUNIT                                            
      U(2,2,NZ)=-SCALAR*IUNIT                                           
      U(2,NY,NZ)=SCALAR*IUNIT                                           
      V(2,2,2)=SCALAR*IUNIT                                             
      V(2,NY,2)=SCALAR*IUNIT                                            
      V(2,2,NZ)=SCALAR*IUNIT                                            
      V(2,NY,NZ)=SCALAR*IUNIT                                           
      RETURN                                                            
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE  WCAL (U,V,W,OX,OY,OZ)                                 
C                                                                       
C
C     COMPUTE THE VORTICITY VECTOR W = (OX,OY,OZ)
C                                    = (XW,YW,ZW) X (U,V,W)   
C
C     ON ENTRY
C
C     U, V, W
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 U(NXHP,NY,NZ),V(NXHP,NY,NZ),W(NXHP,NY,NZ)
C
C     ON RETURN
C
C     OX, OY, OZ
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 OX(NXHP,NY,OZ),OY(NXHP,NY,OZ),OZ(NXHP,NY,OZ)
C
      IMPLICIT REAL*8(A-H,O-Z)                                          
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      COMMON/WAV/ XW(129),YW(129),ZW(129),XSQ(129),YSQ(129),ZSQ(129)    
      COMPLEX*16 U(NXHP,NY,1),V(NXHP,NY,1),W(NXHP,NY,1),                
     +     OX(NXHP,NY,1),OY(NXHP,NY,1),OZ(NXHP,NY,1),IUNIT              
      IUNIT = CMPLX(0.D0,1.D0)                                          
      DO 1 J=1,NY                                                       
         Q = YW(J)                                                      
         DO 2 I=1,NXHP                                                  
            P = XW(I)                                                   
            DO 3 K=1,NZ                                                 
               OX(I,J,K) = IUNIT*(Q*W(I,J,K) - ZW(K)*V(I,J,K))          
               OY(I,J,K) = IUNIT*(ZW(K)*U(I,J,K) - P*W(I,J,K))          
               OZ(I,J,K) = IUNIT*(P*V(I,J,K) - Q*U(I,J,K))              
    3       CONTINUE                                                    
    2    CONTINUE                                                       
    1 CONTINUE                                                          
      RETURN                                                            
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE DCOPY(N,X,IX,Y,IY)                                     
C                                                                       
C     COPY A VECTOR X OF LENGTH N, TO ANOTHER VECTOR Y OF LENGTH N:     
C     Y <- X                                                            
C                                                                       
C
C     ON ENTRY
C
C     N
C        INTEGER ,IS THE NUMBER OF ELEMENTS IN VECTORS X AND Y. N >= 0.
C
C     X
C        REAL*8 X(*), IS THE VECTOR OF LENGTH N. 
C        X IS A ONE-DIMENSIONAL ARRAY OF (AT LEAST LENGTH) 1+(N-1)|INCX.
C
C     INCX
C        INTEGER, IS THE STRIDE FOR VECTOR X.
C
C     INCY
C        INTEGER, IS THE STRIDE FOR VECTOR Y.
C
C     ON RETURN
C
C     Y
C        REAL*8 Y(*), IS THE VECTOR OF LENGTH N. 
C        Y IS A ONE-DIMENSIONAL ARRAY OF (AT LEAST LENGTH) 1+(N-1)|INCY.
C
C     X IS A ONE-DIMENSIONAL ARRAY OF (AT LEAST LENGTH) 1+(N-1)|INCX|   
C     Y IS A ONE-DIMENSIONAL ARRAY OF (AT LEAST LENGTH) 1+(N-1)|INCY|   
C                                                                       
      INTEGER N,IX,IY                                                   
      REAL*8  X(*), Y(*)                                                
C                                                                       
      INTEGER I                                                         
C                                                                       
      IF ((IX.EQ.1).AND.(IY.EQ.1)) THEN                                 
         DO 10 I=1,N                                                    
            Y(I)=X(I)                                                   
   10    CONTINUE                                                       
         ELSE                                                           
         DO 20 I=0,N-1                                                  
            Y(I*IY+1)=X(I*IX+1)                                         
   20    CONTINUE                                                       
      ENDIF                                                             
C                                                                       
      RETURN                                                            
      END
C                                                                       
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE  ENR(A,B,COEF,S)                                       
C                                                                       
C     COMPUTES VECTOR MULTIPLY $2*COEF*A*B$                             
C
C     ON ENTRY
C   
C     A, B
C        ARE FOUR DIMENSIONAL ARRAYS.
C        REAL*8 A(2,NXHP,NY,NZ),B(2,NXHP,NY,NZ)
C
C     COEF
C        IS A CONSTANT.
C        REAL*8 COEF.
C
C     ON RETURN
C
C     S
C        COMPUTES VECTOR MULTIPLY.
C        REAL*8 S.
      IMPLICIT REAL*8(A-H,O-Z)                                          
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      REAL*8        A(2,NXHP,NY,1),B(2,NXHP,NY,1)                       
      REAL*8 S0
      S = 0.D0                                                          
        S0=0
      DO 1 IA=1,2                                                       
         DO 2 J=1,NY                                                    
            DO 3 I=1,NXHP-1                                             
               IF (I.EQ.1) THEN                                         
                  F1 = 1.D0                                             
                  ELSE                                                  
                  F1 = 2.D0                                             
               END IF                                                   
               DO 4 K=1,NZ                                              
              S0=S0+F1*COEF*(A(IA,I,J,K)*B(IA,I,J,K))
    4          CONTINUE                                                 
    3       CONTINUE                                                    
    2    CONTINUE                                                       
    1 CONTINUE                                                          
              S=S+S0
      RETURN                                                            
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE  ZFFT(JN,NTSK,IND,IS)                                  
C
C     PERFORMS 3-D FOURIER TRANSFORMS OF COMPLEX (CFT) 
C     DATA IN Z DIRECTION.
C
C     SUBROUTINE CALL: DCFT
C
C     ON ENTRY
C
C     JN
C        INTEGER, IS AN INDEX OF START POINT IN THE ARRAY.
C
C     IND
C        INTEGER, IS AN INDEX TO POINT OUT WHICH ARRAY SHOULD 
C        APPLY DCFT ROUTINE.
C
C     IS 
C        INTEGER, CONTROLS THE DIRECTION OF THE TRANSFORM. 
C        IS EITHER = 1 OR -1.
C        IF IS = 1, TRANSFORMING FREQUENCY TO TIME.
C        IF IS = -1, TRANSFORMING TIME TO FREQUENCY.
C
      IMPLICIT    REAL*8(A-H,O-Z)                                       
      PARAMETER   (IX= 64,IY= 64,IZ= 64 ,M1=6)

      PARAMETER   (IXPP=IX+2,IXHP=IX/2+1,ITOT=IXPP*IY*IZ,               
     +          IXY=IXHP*IY,IXZ=IXHP*IZ,IYZ=IY*IZ)                      
      COMMON/ALL/ U(IXPP,IY,IZ),V(IXPP,IY,IZ),W(IXPP,IY,IZ),            
     +            OX(IXPP,IY,IZ),OY(IXPP,IY,IZ),OZ(IXPP,IY,IZ)          
      COMMON/SHR/ U0(IXPP,IY,IZ),V0(IXPP,IY,IZ),W0(IXPP,IY,IZ),         
     +            U1(IXPP,IY,IZ),V1(IXPP,IY,IZ),W1(IXPP,IY,IZ)          
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      REAL*8      AUX1(4*IX),AUX2(2*IX)                               
      IF(IS.EQ.-1) THEN                                                 
         IF(IND.EQ.1) THEN                                              
            SZ = 1.D0/FLOAT(NZ)                                         
            CALL DCFT(M1,U(1,JN,1),NXY,1,U(1,JN,1),NXY,1,NZ, NXHP,1,SZ,  
     +      AUX1,AUX2,JN)                                      
         ENDIF                                                          
         IF(IND.EQ.2) THEN                                              
            SZ = 1.D0/FLOAT(NZ)                                         
            CALL DCFT(M1,V(1,JN,1),NXY,1,V(1,JN,1),NXY,1,NZ, NXHP,1,SZ,  
     +      AUX1,AUX2,JN)                                      
         ENDIF                                                          
         IF(IND.EQ.3) THEN                                              
            SZ = 1.D0/FLOAT(NZ)                                         
            CALL DCFT(M1,W(1,JN,1),NXY,1,W(1,JN,1),NXY,1,NZ, NXHP,1,SZ,  
     +      AUX1,AUX2,JN)                                      
         ENDIF                                                          
         IF(IND.EQ.4) THEN                                              
            SZ = 1.D0/FLOAT(NZ)                                         
            CALL DCFT(M1,OX(1,JN,1),NXY,1,OX(1,JN,1),NXY,1,NZ,NXHP,1,SZ,
     +      AUX1,AUX2,JN)                                      
         ENDIF                                                          
         IF(IND.EQ.5) THEN                                              
            SZ = 1.D0/FLOAT(NZ)                                         
            CALL DCFT(M1,OY(1,JN,1),NXY,1,OY(1,JN,1),NXY,1,NZ,NXHP,1,SZ,
     +      AUX1,AUX2,JN)                                      
         ENDIF                                                          
         IF(IND.EQ.6) THEN                                              
            SZ = 1.D0/FLOAT(NZ)                                         
            CALL DCFT(M1,OZ(1,JN,1),NXY,1,OZ(1,JN,1),NXY,1,NZ,NXHP,1,SZ,
     +      AUX1,AUX2,JN)                                      
         ENDIF                                                          
         ELSE                                                           
         IF(IND.EQ.1) THEN                                              
            CALL DCFT(M1,U(1,JN,1),NXY,1,U(1,JN,1),NXY,1,NZ, NXHP,-1,1.  
     +      0D0,AUX1,AUX2,JN)                                  
         ENDIF                                                          
         IF(IND.EQ.2) THEN                                              
            CALL DCFT(M1,V(1,JN,1),NXY,1,V(1,JN,1),NXY,1,NZ, NXHP,-1,1.  
     +      0D0,AUX1,AUX2,JN)                                  
         ENDIF                                                          
         IF(IND.EQ.3) THEN                                              
            CALL DCFT(M1,W(1,JN,1),NXY,1,W(1,JN,1),NXY,1,NZ, NXHP,-1,1.  
     +      0D0,AUX1,AUX2,JN)                                  
         ENDIF                                                          
         IF(IND.EQ.4) THEN                                              
            CALL DCFT(M1,OX(1,JN,1),NXY,1,OX(1,JN,1),NXY,1,NZ,NXHP,-1,1.
     +      0D0,AUX1,AUX2,JN)                                  
         ENDIF                                                          
         IF(IND.EQ.5) THEN                                              
            CALL DCFT(M1,OY(1,JN,1),NXY,1,OY(1,JN,1),NXY,1,NZ,NXHP,-1,1.
     +      0D0,AUX1,AUX2,JN)                                  
         ENDIF                                                          
         IF(IND.EQ.6) THEN                                              
            CALL DCFT(M1,OZ(1,JN,1),NXY,1,OZ(1,JN,1),NXY,1,NZ,NXHP,-1,1.
     +      0D0,AUX1,AUX2,JN)                                  
         ENDIF                                                          
      ENDIF                                                             
      RETURN                                                            
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE DCFT( M1, X, INC1X, INC2X, Y, INC1Y, INC2Y, N, M,
     &   ISIGN, SCALE, RA, AUX,NQ)
C
C  THIS PERFORMS A SET OF M COMPLEX DISCRETE N-POINT FOURIER TRANSFORMS
C  OF COMPLEX DATA. HERE N=2^M1.
C
C     SUBROUTINE CALL: CFFT
C
C     ON ENTRY
C 
C     M1, N
C        INTEGER IRUN, N1. N1 IS THE LENGTH OF EACH SEQUENCE TO 
C        BE TRANSFORMED. WHERE N1 = 2^IRUN. 2 < IRUN < 8.
C     X
C        REAL*8 X(*), CONSISTING OF M SEQUENCES OF LENGTH N.
C
C     INC1X, INC1Y
C        INTEGER, ARE THE STRIDE BETWEEN THE ELEMENTS WITHIN 
C        EACH SEQUENCE IN ARRAY X AND Y.
C
C     INC2X, INC2Y
C        INTEGER, ARE THE STRIDE BETWEEN THE FIRST ELEMENTS 
C        OF SEQUENCE IN ARRAY X AND Y.
C     M 
C        INTEGER. M > 0, IS THE NUMBER OF SEQUENCE TO BE TRAMSFORMED.
C
C     ISIGN
C        INTEGER, CONTROLS THE DIRECTION OF THE TRANSFORM. 
C        ISIGN  EITHER = 1 OR -1.
C        IF IS = -1, TRANSFORMING FREQUENCY TO TIME.
C        IF IS = 1, TRANSFORMING TIME TO FREQUENCY.
C      
C     SCALE
C        REAL*8, IS THE SCALING CONSTANT SCALE. SCALE<>0.0
C
C     RA, AUX
C        REAL*8 RA(4*N), AUX(2*N), ARE THE SCRATCH ARRAY.
C
C     ON RETURN
C
C     Y
C        REAL*8 Y(*), CONTAINS THE RESULTS OF THE M DISCRETE 
C        FOURIER TRANSFORMS, EACH OF LENGTH N.
C
      IMPLICIT REAL*8(A-H,O-Z)
      REAL*8 X(*),Y(*),RA(4*N),AUX(2*N)
          CALL CFFT(0,M1,AUX,RA,RA(2*N+1),NQ)
        DO 10 I=1,M
          IBR = (I-1)*INC2X*2 + 1
          IBC = (I-1)*INC2X*2 + 2
          DO 20 II=1,N
            INDR = IBR + (II-1)*2*INC1X
            INDC = IBC + (II-1)*2*INC1X
            RA(II) = X(INDR)
            RA(II+N) = X(INDC)
20        CONTINUE
          CALL CFFT(ISIGN,M1,AUX,RA,RA(2*N+1),NQ)
          DO 30 II=1,N
            INDR = IBR + (II-1)*2*INC1X
            INDC = IBC + (II-1)*2*INC1X
             Y(INDR) = SCALE*RA(II)
             Y(INDC) = SCALE*RA(II+N)
30         CONTINUE
10     CONTINUE
      RETURN
      END
C
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE CFFT (IS, M, U, X, Y, NQ)
C
C   Computes the 2^M-point complex-to-complex FFT of X using an algorithm due
C   to Swarztrauber, coupled with some fast methods for performing power-of-
C   two matrix transpositions (see article by DHB in Intl. J. of Supercomputer
C   Applications, Spring 1988, p. 82 - 87). This is the radix 2 version.
C   X is both the input and the output array, while Y is a scratch array.
C   Both X and Y must be dimensioned with 2 * N real cells, where N = 2^M.
C   The data in X are assumed to have real and imaginary parts separated
C   by N cells.  Before calling CFFTZ to perform an FFT, the array U must be
C   initialized by calling CFFTZ with IS set to 0 and M set to MX, where MX is
C   the maximum value of M for any subsequent call.  U must be dimensioned
C   with at least 2 * NX real cells, where NX = 2^MX.
C
C   David H. Bailey     October 26, 1990
C
      IMPLICIT REAL*8 (A-H, O-Z)
      PARAMETER (PI = 3.141592653589793238D0)
      COMMON /COUNT/ KTTRANS(256)
      DIMENSION U(1), X(1), Y(1)
C
      IF (IS .EQ. 0)  THEN
C
C   Initialize the U array with sines and cosines in a manner that permits
C   stride one access at each FFT iteration.
C
        N = 2 ** M
        NU = N
        U(1) = 64 * N + M
        KU = 2
        KN = KU + NU
        LN = 1
C
        DO 110 J = 1, M
          T = PI / LN
C
C   This loop is vectorizable.
C
          DO 100 I = 0, LN - 1
            TI = I * T
            U(I+KU) = COS (TI)
            U(I+KN) = SIN (TI)
 100      CONTINUE
C
          KU = KU + LN
          KN = KU + NU
          LN = 2 * LN
 110    CONTINUE
C
        RETURN
      ENDIF
C
C   Check if input parameters are invalid.
C
      K = U(1)
      MX = MOD (K, 64)
      IF ((IS .NE. 1 .AND. IS .NE. -1) .OR. M .LT. 1 .OR. M .GT. MX)    
     $  THEN
C       WRITE (6, 1)  IS, M, MX
 1      FORMAT ('CFFTZ: EITHER U HAS NOT BEEN INITIALIZED, OR ELSE'/    
     $    'ONE OF THE INPUT PARAMETERS IS INVALID', 3I5)
C       STOP
      ENDIF
C>>
C   A normal call to CFFTZ starts here.  M1 is the number of the first variant
C   radix-2 Stockham iterations to be performed.  The second variant is faster
C   on most computers after the first few iterations, since in the second
C   variant it is not necessary to access roots of unity in the inner DO loop.
C   Thus it is most efficient to limit M1 to some value.  For many vector
C   computers, the optimum limit of M1 is 6.  For scalar systems, M1 should
C   probably be limited to 2.
C
      N = 2 ** M
C      M1 = MIN (M / 2, 6)
      M1 = MIN (M / 2, 2)
      M2 = M - M1
      N2 = 2 ** M1
      N1 = 2 ** M2
C
C   Perform one variant of the Stockham FFT.
C
      DO 120 L = 1, M1, 2
        CALL FFTZ1 (IS, L, M, U, X, Y)
        IF (L .EQ. M1) GOTO 140
        CALL FFTZ1 (IS, L + 1, M, U, Y, X)
 120  CONTINUE
C
C   Perform a transposition of X treated as a N2 x N1 x 2 matrix.
C
      CALL TRANS (N1, N2, X, Y)
      KTTRANS(NQ) = KTTRANS(NQ) + 1
C
C   Perform second variant of the Stockham FFT from Y to X and X to Y.
C
      DO 130 L = M1 + 1, M, 2
        CALL FFTZ2 (IS, L, M, U, Y, X)
        IF (L .EQ. M) GOTO 180
        CALL FFTZ2 (IS, L + 1, M, U, X, Y)
 130  CONTINUE
C
      GOTO 160
C
C   Perform a transposition of Y treated as a N2 x N1 x 2 matrix.
C
 140  CALL TRANS (N1, N2, Y, X)
      KTTRANS(NQ) = KTTRANS(NQ) + 1
C
C   Perform second variant of the Stockham FFT from X to Y and Y to X.
C
      DO 150 L = M1 + 1, M, 2
        CALL FFTZ2 (IS, L, M, U, X, Y)
        IF (L .EQ. M) GOTO 160
        CALL FFTZ2 (IS, L + 1, M, U, Y, X)
 150  CONTINUE
C
      GOTO 180
C
C   Copy Y to X.
C
 160  CONTINUE
      DO 170 I = 1, 2 * N
        X(I) = Y(I)
 170  CONTINUE
C
 180  CONTINUE
      RETURN
      END
C
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE FFTZ1 (IS, L, M, U, X, Y)
C
C
C   PERFORMS THE L-TH ITERATION OF THE FIRST VARIANT OF THE  
C   STOCKHAM FFT.
C
C 
C     ON ENTRY
C
C     IS 
C        INTEGER, CONTROLS THE DIRECTION OF THE TRANSFORM. 
C        IS EITHER = 1 OR -1.
C        IF IS = -1, TRANSFORMING FREQUENCY TO TIME.
C        IF IS = 1, TRANSFORMING TIME TO FREQUENCY.
C     L 
C        INTEGER, L-TH ITERATION OF THE FIRST VARIANT OF THE  
C        STOCKHAM FFT.
C     X
C        REAL*8 X(2*N). X HAS REAL AND IMAGINARY PARTS SEPARATED 
C        BY N CELLS.
C     U
C        REAL*8 U(2*N) IS THE SCRATCH ARRAY.
C     M 
C        INTEGER.  N = 2^M.
C
C     ON RETURN
C
C     Y
C        REAL*8 Y(2*N).
C
      IMPLICIT REAL*8 (A-H, O-Z)
      DIMENSION U(1), X(1), Y(1)
C
C   Set initial parameters.
C
      N = 2 ** M
      K = U(1)
      NU = K / 64
      N1 = N / 2
      LK = 2 ** (L - 1)
      LI = 2 ** (M - L)
      LJ = 2 * LI
      KU = LI + 1
      KN = KU + NU
C
      DO 100 K = 0, LK - 1
        I11 = K * LJ + 1
        I12 = I11 + LI
        I21 = K * LI + 1
        I22 = I21 + N1
C
C   This loop is vectorizable.
C
        DO 100 I = 0, LI - 1
          U1 = U(KU+I)
          U2 = IS * U(KN+I)
          X11 = X(I11+I)
          X12 = X(I11+I+N)
          X21 = X(I12+I)
          X22 = X(I12+I+N)
          T1 = X11 - X21
          T2 = X12 - X22
          Y(I21+I) = X11 + X21
          Y(I21+I+N) = X12 + X22
          Y(I22+I) = U1 * T1 - U2 * T2
          Y(I22+I+N) = U1 * T2 + U2 * T1
 100  CONTINUE
C
      RETURN
      END
C
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE TRANS (N1, N2, X, Y)
C
C   Performs a transpose of the vector X, returning the result in Y.  X is
C   treated as a N1 x N2 complex matrix, and Y is treated as a N2 x N1 complex
C   matrix.  The complex data is assumed stored with real and imaginary parts
C   separated by N1 x N2 locations.  If this routine is to be used for an
C   application involving only real data, then the second line of all inner DO
C   loops may be deleted.
C
C   David H. Bailey      April 28, 1987
C
      IMPLICIT REAL*8 (A-H, O-Z)
      DIMENSION X(1), Y(1)
C
      N = N1 * N2
C>>
C   Perform one of three techniques, depending on N.  The best strategy varies
C   with the computer system.  The following strategy is best for many vector
C   systems.  The outer IF block should be commented out for scalar computers.
C
C      IF (N1 .LT. 32 .OR. N2 .LT. 32) THEN
      IF (N1 .GE. N2) THEN
        GOTO 100
      ELSE
        GOTO 120
      ENDIF
C      ELSE
C        GOTO 140
C      ENDIF
C
C   Scheme 1:  Perform a simple transpose in the usual way.  This is usually
C   the best on vector computers if N2 is odd, or if both N1 and N2 are small,
C   and N1 is larger than N2.
C
 100  DO 110 J = 0, N2 - 1
C
C   This loop is vectorizable.
C
        DO 110 I = 0, N1 - 1
          Y(I*N2+J+1) = X(I+J*N1+1)
          Y(I*N2+J+1+N) = X(I+J*N1+1+N)
 110  CONTINUE
C
      GOTO 180
C
C   Scheme 2:  Perform a simple transpose with the loops reversed.  This is
C   usually the best on vector computers if N1 is odd, or if both N1 and N2 are
C   small, and N2 is larger than N1.
C
 120  DO 130 I = 0, N1 - 1
C
C   This loop is vectorizable.
C
        DO 130 J = 0, N2 - 1
          Y(J+I*N2+1) = X(J*N1+I+1)
          Y(J+I*N2+1+N) = X(J*N1+I+1+N)
 130  CONTINUE
C
      GOTO 180
C
C   Scheme 3:  Perform the transpose along diagonals to insure odd strides.
C   This works well on moderate vector, variable stride computers, when both
C   N1 and N2 are divisible by reasonably large powers of two.
C
 140  N11 = N1 + 1
      N21 = N2 + 1
      IF (N1 .GE. N2) THEN
        K1 = N1
        K2 = N2
        I11 = N1
        I12 = 1
        I21 = 1
        I22 = N2
      ELSE
        K1 = N2
        K2 = N1
        I11 = 1
        I12 = N2
        I21 = N1
        I22 = 1
      ENDIF
C
      DO 150 J = 0, K2 - 1
        J1 = J * I11 + 1
        J2 = J * I12 + 1
C
C   This loop is vectorizable.
C
        DO 150 I = 0, K2 - 1 - J
          Y(N21*I+J2) = X(N11*I+J1)
          Y(N21*I+J2+N) = X(N11*I+J1+N)
 150  CONTINUE
C
      DO 160 J = 1, K1 - K2 - 1
        J1 = J * I21 + 1
        J2 = J * I22 + 1
C
C   This loop is vectorizable.
C
        DO 160 I = 0, K2 - 1
          Y(N21*I+J2) = X(N11*I+J1)
          Y(N21*I+J2+N) = X(N11*I+J1+N)
 160  CONTINUE
C
      DO 170 J = K1 - K2, K1 - 1
        J1 = J * I21 + 1
        J2 = J * I22 + 1
C
C   This loop is vectorizable.
C
        DO 170 I = 0, K1 - 1 - J
          Y(N21*I+J2) = X(N11*I+J1)
          Y(N21*I+J2+N) = X(N11*I+J1+N)
 170  CONTINUE
C
 180  RETURN
      END
C
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE FFTZ2 (IS, L, M, U, X, Y)
C
C   PERFORMS THE L-TH ITERATION OF THE SECOND VARIANT OF THE  
C   STOCKHAM FFT.
C
C 
C     ON ENTRY
C
C     IS 
C        INTEGER, CONTROLS THE DIRECTION OF THE TRANSFORM. 
C        IS EITHER = 1 OR -1.
C        IF IS = -1, TRANSFORMING FREQUENCY TO TIME.
C        IF IS = 1, TRANSFORMING TIME TO FREQUENCY.
C     L 
C        INTEGER, L-TH ITERATION OF THE SECOND VARIANT OF THE  
C        STOCKHAM FFT.
C     X
C        REAL*8 X(2*N). X HAS REAL AND IMAGINARY PARTS SEPARATED 
C        BY N CELLS.
C     U
C        REAL*8 U(2*N) IS THE SCRATCH ARRAY.
C     M 
C        INTEGER.  N = 2^M.
C
C     ON RETURN
C
C     Y
C        REAL*8 Y(2*N).
C
      IMPLICIT REAL*8 (A-H, O-Z)
      DIMENSION U(1), X(1), Y(1)
C
C   Set initial parameters.
C
      N = 2 ** M
      K = U(1)
      NU = K / 64
      N1 = N / 2
      LK = 2 ** (L - 1)
      LI = 2 ** (M - L)
      LJ = 2 * LK
      KU = LI + 1
C
      DO 100 I = 0, LI - 1
        I11 = I * LK + 1
        I12 = I11 + N1
        I21 = I * LJ + 1
        I22 = I21 + LK
        U1 = U(KU+I)
        U2 = IS * U(KU+I+NU)
C
C   This loop is vectorizable.
C
        DO 100 K = 0, LK - 1
          X11 = X(I11+K)
          X12 = X(I11+K+N)
          X21 = X(I12+K)
          X22 = X(I12+K+N)
          T1 = X11 - X21
          T2 = X12 - X22
          Y(I21+K) = X11 + X21
          Y(I21+K+N) = X12 + X22
          Y(I22+K) = U1 * T1 - U2 * T2
          Y(I22+K+N) = U1 * T2 + U2 * T1
 100  CONTINUE
C
      RETURN
      END
C
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE  XYFFT(KN,NTSK,IND,IS)                                 
C
C     PERFORMS 3-D FOURIER TRANSFORMS OF COMPLEX (CFT) 
C     DATA IN X, Y DIRECTION.
C
C     SUBROUTINE CALLS: DCFT, DCRFT, and DRCFT.
C
C     ON ENTRY
C
C     KN
C        INTEGER, IS AN INDEX OF START POINT IN THE ARRAY.
C
C     IND
C        INTEGER, IS AN INDEX TO POINT OUT WHICH ARRAY SHOULD 
C        APPLY DCFT ROUTINE.
C
C     IS
C        INTEGER, CONTROLS THE DIRECTION OF THE TRANSFORM. 
C        IS EITHER = 1 OR -1.
C        IF IS = 1, TRANSFORMING FREQUENCY TO TIME.
C        IF IS = -1, TRANSFORMING TIME TO FREQUENCY.
C
      IMPLICIT    REAL*8(A-H,O-Z)                                       
      PARAMETER   (IX= 64,IY= 64,IZ= 64 ,M1=6)

      PARAMETER   (IXPP=IX+2,IXHP=IX/2+1,ITOT=IXPP*IY*IZ,               
     +          IXY=IXHP*IY,IXZ=IXHP*IZ,IYZ=IY*IZ)                      
      COMMON/ALL/ U(IXPP,IY,IZ),V(IXPP,IY,IZ),W(IXPP,IY,IZ),            
     +            OX(IXPP,IY,IZ),OY(IXPP,IY,IZ),OZ(IXPP,IY,IZ)          
      COMMON/SHR/ U0(IXPP,IY,IZ),V0(IXPP,IY,IZ),W0(IXPP,IY,IZ),         
     +            U1(IXPP,IY,IZ),V1(IXPP,IY,IZ),W1(IXPP,IY,IZ)          
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      REAL*8      AUX1(4*IX),AUX2(2*IX)                               
      IF(IS.EQ.-1) THEN                                                 
         IF(IND.EQ.1) THEN                                              
            SX = 1.D0/FLOAT(NX)                                         
            CALL DRCFT(M1,U(1,1,KN),NXPP,U(1,1,KN),NXHP,NX,NY,1,SX,      
     +      AUX1,AUX2,KN)                                      
            SY = 1.D0/FLOAT(NY)                                         
            CALL DCFT(M1,U(1,1,KN),NXHP,1,U(1,1,KN),NXHP,1,NY,NXHP,1,SY,
     +      AUX1,AUX2,KN)                                      
         ENDIF                                                          
         IF(IND.EQ.2) THEN                                              
            SX = 1.D0/FLOAT(NX)                                         
            CALL DRCFT(M1,V(1,1,KN),NXPP,V(1,1,KN),NXHP,NX,NY,1,SX,      
     +      AUX1,AUX2,KN)                                      
            SY = 1.D0/FLOAT(NY)                                         
            CALL DCFT(M1,V(1,1,KN),NXHP,1,V(1,1,KN),NXHP,1,NY,NXHP,1,SY,
     +      AUX1,AUX2,KN)                                      
         ENDIF                                                          
         IF(IND.EQ.3) THEN                                              
            SX = 1.D0/FLOAT(NX)                                         
            CALL DRCFT(M1,W(1,1,KN),NXPP,W(1,1,KN),NXHP,NX,NY,1,SX,      
     +      AUX1,AUX2,KN)                                      
            SY = 1.D0/FLOAT(NY)                                         
            CALL DCFT(M1,W(1,1,KN),NXHP,1,W(1,1,KN),NXHP,1,NY,NXHP,1,SY,
     +      AUX1,AUX2,KN)                                      
         ENDIF                                                          
         IF(IND.EQ.4) THEN                                              
            SX = 1.D0/FLOAT(NX)                                         
            CALL DRCFT(M1,OX(1,1,KN),NXPP,OX(1,1,KN),NXHP,NX,NY,1,SX,    
     +      AUX1,AUX2,KN)                                      
            SY = 1.D0/FLOAT(NY)                                         
            CALL DCFT(M1,OX(1,1,KN),NXHP,1,OX(1,1,KN),NXHP,1,NY, NXHP,1, 
     +      SY,AUX1,AUX2,KN)                                   
         ENDIF                                                          
         IF(IND.EQ.5) THEN                                              
            SX = 1.D0/FLOAT(NX)                                         
            CALL DRCFT(M1,OY(1,1,KN),NXPP,OY(1,1,KN),NXHP,NX,NY,1,SX,    
     +      AUX1,AUX2,KN)                                      
            SY = 1.D0/FLOAT(NY)                                         
            CALL DCFT(M1,OY(1,1,KN),NXHP,1,OY(1,1,KN),NXHP,1,NY, NXHP,1, 
     +      SY,AUX1,AUX2,KN)                                   
         ENDIF                                                          
         IF(IND.EQ.6) THEN                                              
            SX = 1.D0/FLOAT(NX)                                         
            CALL DRCFT(M1,OZ(1,1,KN),NXPP,OZ(1,1,KN),NXHP,NX,NY,1,SX,    
     +      AUX1,AUX2,KN)                                      
            SY = 1.D0/FLOAT(NY)                                         
            CALL DCFT(M1,OZ(1,1,KN),NXHP,1,OZ(1,1,KN),NXHP,1,NY, NXHP,1, 
     +      SY,AUX1,AUX2,KN)                                   
         ENDIF                                                          
         ELSE                                                           
         IF(IND.EQ.1) THEN                                              
            CALL DCFT(M1,U(1,1,KN),NXHP,1,U(1,1,KN),NXHP,1,NY, NXHP,-1,
     +      1.D0,AUX1,AUX2,KN)                                   
            CALL DCRFT(M1,U(1,1,KN),NXHP,U(1,1,KN),NXPP,NX,NY,-1,1.D0,   
     +      AUX1,AUX2,KN)                                      
         ENDIF                                                          
         IF(IND.EQ.2) THEN                                              
            CALL DCFT(M1,V(1,1,KN),NXHP,1,V(1,1,KN),NXHP,1,NY, NXHP,-1,
     +      1.D0,AUX1,AUX2,KN)                                   
            CALL DCRFT(M1,V(1,1,KN),NXHP,V(1,1,KN),NXPP,NX,NY,-1,1.D0,   
     +      AUX1,AUX2,KN)                                      
         ENDIF                                                          
         IF(IND.EQ.3) THEN                                              
            CALL DCFT(M1,W(1,1,KN),NXHP,1,W(1,1,KN),NXHP,1,NY, NXHP,-1,
     +      1.D0,AUX1,AUX2,KN)                                   
            CALL DCRFT(M1,W(1,1,KN),NXHP,W(1,1,KN),NXPP,NX,NY,-1,1.D0,   
     +      AUX1,AUX2,KN)                                      
         ENDIF                                                          
         IF(IND.EQ.4) THEN                                              
            CALL DCFT(M1,OX(1,1,KN),NXHP,1,OX(1,1,KN),NXHP,1,NY,NXHP,-1,
     +      1.D0,AUX1,AUX2,KN)                                 
            CALL DCRFT(M1,OX(1,1,KN),NXHP,OX(1,1,KN),NXPP,NX,NY,-1,1.D0, 
     +      AUX1,AUX2,KN)                                      
         ENDIF                                                          
         IF(IND.EQ.5) THEN                                              
            CALL DCFT(M1,OY(1,1,KN),NXHP,1,OY(1,1,KN),NXHP,1,NY,NXHP,-1,
     +      1.D0,AUX1,AUX2,KN)                                 
            CALL DCRFT(M1,OY(1,1,KN),NXHP,OY(1,1,KN),NXPP,NX,NY,-1,1.D0, 
     +      AUX1,AUX2,KN)                                      
         ENDIF                                                          
         IF(IND.EQ.6) THEN                                              
            CALL DCFT(M1,OZ(1,1,KN),NXHP,1,OZ(1,1,KN),NXHP,1,NY,NXHP,-1,
     +      1.D0,AUX1,AUX2,KN)                                 
            CALL DCRFT(M1,OZ(1,1,KN),NXHP,OZ(1,1,KN),NXPP,NX,NY,-1,1.D0, 
     +      AUX1,AUX2,KN)                                      
         ENDIF                                                          
      ENDIF                                                             
      RETURN                                                            
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE DRCFT( IRUN, X, INC2X, Y, INC2Y, N1, M1, ISIGN,        
     +   SCALE, DUMMY1,DUMMY2,NQ)                                   
C 
C     PERFORMS THE COMPLEX TO REAL PASS AS PART OF CFT ROUTINE.
C
C     SUBROUTINE CALL: DCFT.
C
C     ON ENTRY
C 
C     IRUN, N1
C        INTEGER IRUN, N1. WHERE N1 = 2^IRUN. 2 < IRUN < 8.
C     X
C        REAL*8 X(INC2X,*), IS THE TWO DIMENSION ARRAY.
C     Y
C        REAL*8 Y(2,INC2Y,*), IS THE THREE DIMENSION ARRAY.
C
C     INC2X, INC2Y
C        INTEGER, ARE THE STRIDE BETWEEN THE ELEMENTS WITHIN EACH 
C        SEQUENCE IN ARRAY X AND Y.
C     M1 
C        INTEGER. M1 > 0. IS THE NUMBER OF SEQUENCE TO BE TRAMSFORMED.
C
C     ISIGN
C        INTEGER, CONTROLS THE DIRECTION OF THE TRANSFORM. 
C        ISIGN  EITHER = 1 OR -1.
C        IF IS = -1, TRANSFORMING FREQUENCY TO TIME.
C        IF IS = 1, TRANSFORMING TIME TO FREQUENCY.
C      
C     SCALE
C        REAL*8, IS THE SCALING CONSTANT SCALE. SCALE<>0.0
C
C     DUMMY1, DUMMY2
C        REAL*8 DUMMY1(*), DUMMY2(*), ARE THE SCRATCH ARRAY.
C
      REAL*8 X(INC2X,*),Y(2,INC2Y,*),DUMMY1(*),DUMMY2(*)                
      REAL*8 PI,SCALE, PJ,PR,FR,FI,GR,GI,TR,TI                           
      REAL*8 PIHALF,ALPHA                                               
      PARAMETER ( PI=3.141592653589793D0 )                              
      NHALF = N1/2                                                      
      CALL DCFT(IRUN-1,X,1,INC2X/2,Y,1,INC2Y,N1/2,M1,ISIGN,         
     +      SCALE, DUMMY1,DUMMY2,NQ)                                     
      J2=NHALF+1                                                        
      DO 1 I = 1 , M1                                                   
         Y(1,J2,I) = Y(1,1,I) - Y(2,1,I)                                
         Y(1,1,I) = Y(1,1,I) + Y(2,1,I)                                 
         Y(2,1,I) = 0.D0                                                
         Y(2,J2,I) = 0.D0                                               
    1 CONTINUE                                                          
      PIHALF = PI/NHALF                                                 
      DO 2 J = 1 , NHALF/2                                              
         ALPHA = J * PIHALF                                             
         PJ = -ISIGN * SIN( ALPHA )                                     
         PR = - COS( ALPHA )                                            
         J1=J+1                                                         
         J2=NHALF-J+1                                                   
         DO 3 I = 1 , M1                                                
            FR = Y(1,J1,I) + Y(1,J2,I)                                  
            FI = Y(2,J1,I) - Y(2,J2,I)                                  
            TR = Y(1,J1,I) - Y(1,J2,I)                                  
            TI = Y(2,J1,I) + Y(2,J2,I)                                  
            GR = PJ * TR - PR * TI                                      
            GI = PJ * TI + PR * TR                                      
            Y(1,J1,I) = ( FR + GR ) * .5D0                              
            Y(2,J1,I) = ( FI + GI ) * .5D0                              
            Y(1,J2,I) = ( FR - GR ) * .5D0                              
            Y(2,J2,I) = -( FI - GI ) * .5D0                             
    3    CONTINUE                                                       
    2 CONTINUE                                                          
      RETURN                                                            
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE DCRFT(IRUN,X,INC2X,Y,INC2Y,N1,M1,ISIGN,SCALE,DUMMY1,   
     + DUMMY2,NQ)                                                  
C 
C     PERFORMS THE REAL TO COMPLEX PASS AS PART OF CFT ROUTINE.
C
C     SUBROUTINE CALL: DCFT.
C
C     ON ENTRY
C 
C     IRUN, N1
C        INTEGER IRUN, N1. WHERE N1 = 2^IRUN. 2 < IRUN < 8.
C     X
C        REAL*8 X(INC2X,*), IS THE TWO DIMENSION ARRAY.
C     Y
C        REAL*8 Y(2,INC2Y,*), IS THE THREE DIMENSION ARRAY.
C
C     INC2X, INC2Y
C        INTEGER, ARE THE STRIDE BETWEEN THE ELEMENTS WITHIN EACH 
C        SEQUENCE IN ARRAY X AND Y.
C     M1 
C        INTEGER. M1 > 0. IS THE NUMBER OF SEQUENCE TO BE TRAMSFORMED.
C
C     ISIGN
C        INTEGER, CONTROLS THE DIRECTION OF THE TRANSFORM. 
C        ISIGN  EITHER = 1 OR -1.
C        IF IS = -1, TRANSFORMING FREQUENCY TO TIME.
C        IF IS = 1, TRANSFORMING TIME TO FREQUENCY.
C      
C     SCALE
C        REAL*8, IS THE SCALING CONSTANT SCALE. SCALE <> 0.0
C
C     DUMMY1, DUMMY2
C        REAL*8 DUMMY1(*), DUMMY2(*), ARE THE SCRATCH ARRAY.
C
      REAL*8 Y(INC2Y,*), X(2,INC2X,*),FR,FI,GR,GI,TR,TI,PR,PJ,PI        
      REAL*8 DUMMY1(*), DUMMY2(*)                                       
      REAL*8 SCALE, PIHALF, ALPHA                                       
      PARAMETER ( PI=3.141592653589793D0)                               
      NHALF = N1/2                                                      
      J2=NHALF+1                                                        
      DO 3 I = 1 , M1                                                   
         X(2,1,I) = X(1,1,I) - X(1,J2,I)                                
         X(1,1,I) = X(1,1,I) + X(1,J2,I)                                
    3 CONTINUE                                                          
      PIHALF=PI/NHALF                                                   
      DO 4 J = 1 , NHALF/2                                              
         ALPHA = J * PIHALF                                             
         PJ = ISIGN * SIN( ALPHA )                                      
         PR = COS( ALPHA )                                              
         J1=J+1                                                         
         J2=NHALF-J+1                                                   
         DO 5 I = 1 , M1                                                
            FR = X(1,J1,I) + X(1,J2,I)                                  
            FI = X(2,J1,I) - X(2,J2,I)                                  
            TR = X(1,J1,I) - X(1,J2,I)                                  
            TI = X(2,J1,I) + X(2,J2,I)                                  
            GR = PJ * TR - PR * TI                                      
            GI = PJ * TI + PR * TR                                      
            X(1,J1,I) = FR + GR                                         
            X(2,J1,I) = FI + GI                                         
            X(1,J2,I) = FR - GR                                         
            X(2,J2,I) = - ( FI - GI )                                   
    5    CONTINUE                                                       
    4 CONTINUE                                                          
      CALL DCFT(IRUN-1,X,1,INC2X,Y,1,INC2Y/2,N1/2,M1,ISIGN,SCALE,         
     +       DUMMY1,DUMMY2,NQ)                                     
      RETURN                                                            
      END
C                                                                       
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE UXW(A,B,C,D,E,F)                                       
C                                                                       
C                                                                       
C     COMPUTE THE NONLINEAR TERM (D, E, F) = (A, B, C) X (D, E, F)
C
C     ON ENTRY
C  
C     A, B, C
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 A(NXHP,NY,NZ),B(NXHP,NY,NZ),C(NXHP,NY,NZ)
C
C     D, E, F     
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 D(NXHP,NY,NZ),E(NXHP,NY,NZ),F(NXHP,NY,NZ)
C
C     ON RETURN
C 
C     D, E, F     
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 D(NXHP,NY,NZ),E(NXHP,NY,NZ),F(NXHP,NY,NZ)
C                                                                       
      IMPLICIT REAL*8(A-H,O-Z)                                          
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      DIMENSION  A(NXPP,NY,1),B(NXPP,NY,1),C(NXPP,NY,1),                
     +           D(NXPP,NY,1),E(NXPP,NY,1),F(NXPP,NY,1)                 
      DO 1 K=1,NZ                                                       
         DO 2 J=1,NY                                                    
            DO 3 I=1,NXPP                                               
               TA = A(I,J,K)                                            
               TB = B(I,J,K)                                            
               TC = C(I,J,K)                                            
               TEMP1 = TB*F(I,J,K) - TC*E(I,J,K)                        
               TEMP2 = TC*D(I,J,K) - TA*F(I,J,K)                        
               TEMP3 = TA*E(I,J,K) - TB*D(I,J,K)                        
               D(I,J,K) = TEMP1                                         
               E(I,J,K) = TEMP2                                         
               F(I,J,K) = TEMP3                                         
    3       CONTINUE                                                    
    2    CONTINUE                                                       
    1 CONTINUE                                                          
      RETURN                                                            
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC 
C                                                                       
      SUBROUTINE  LINAVG(AM,BM,CM,D,E,F,DT2,DTV,DTT)                    

C     LEAPFROG AVERAGING IS DONE PERIODICALLY TO PREVENT THE 
C     DECOUPLING OF TIME STEPS DURING LONG-TIME INTEGRATION.
C
C     ON ENTRY
C 
C     AM, BM, CM
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 AM(NXHP,NY,NZ),BM(NXHP,NY,NZ),CM(NXHP,NY,NZ)
C
C     D, E, F
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 D(NXHP,NY,NZ),E(NXHP,NY,NZ),F(NXHP,NY,NZ)
C
C     DTT, DT2
C        ARE DELTA TIME, DOUBLE DELTA TIME.
C        REAL*8 DTT, DT2.
C
C     DTV
C        IS THE VISCOSITY COEFFICIENT.
C        REAL*8 DTV.
C
C     ON RETURN
C
C     D, E, F
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 D(NXHP,NY,NZ),E(NXHP,NY,NZ),F(NXHP,NY,NZ)
C
      IMPLICIT REAL*8(A-H,O-Z)                                          
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      COMMON/WAV/ XW(129),YW(129),ZW(129),XSQ(129),YSQ(129),ZSQ(129)    
      COMMON/TRUNC/XWT(129),YWT(129),ZWT(129)                           
      COMPLEX*16  AM(NXHP,NY,1),BM(NXHP,NY,1),CM(NXHP,NY,1),            
     +            D(NXHP,NY,1),E(NXHP,NY,1),F(NXHP,NY,1)                
      COMPLEX*16 P                                                      
      DO 1 K=1,NZ                                                       
         R = ZW(K)                                                      
         RT= ZWT(K)                                                     
         ZWSQ = ZSQ(K)                                                  
         DO 1 J=1,NY                                                    
            Q = YW(J)                                                   
            QT= YWT(J)                                                  
            YWSQ = YSQ(J) + ZWSQ                                        
            DO 2 I=1,NXHP                                               
               FACT3 = XSQ(I) + YWSQ                                    
               FACT1 = DTV*DTT*FACT3                                    
               FACT2 = RT*QT*XWT(I)/(1.D0 + FACT1)                      
               D(I,J,K) = AM(I,J,K)-FACT1*AM(I,J,K) + DT2*D(I,J,K)      
               E(I,J,K) = BM(I,J,K)-FACT1*BM(I,J,K) + DT2*E(I,J,K)      
               F(I,J,K) = CM(I,J,K)-FACT1*CM(I,J,K) + DT2*F(I,J,K)      
               P = (F(I,J,K)*R + Q*E(I,J,K) + XW(I)*D(I,J,K))/FACT3     
               D(I,J,K) = AM(I,J,K) + (D(I,J,K) - XW(I)*P)*FACT2        
               E(I,J,K) = BM(I,J,K) + (E(I,J,K) - Q*P)*FACT2            
               F(I,J,K) = CM(I,J,K) + (F(I,J,K) - R*P)*FACT2            
    2       CONTINUE                                                    
    1 CONTINUE                                                          
      RETURN                                                            
      END
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE  MIXAVG(A,B,C,D,E,F)                                   
C
C     PART OF LEAPFROG AVERAGING IS DONE PERIODICALLY TO PREVENT THE 
C     DECOUPLING OF TIME STEPS DURING LONG-TIME INTEGRATION.
C
C     ON ENTRY
C 
C     A, B, C
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 A(NXHP,NY,NZ),B(NXHP,NY,NZ),C(NXHP,NY,NZ)
C
C     D, E, F
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 D(NXHP,NY,NZ),E(NXHP,NY,NZ),F(NXHP,NY,NZ)
C
C     ON RETURN
C 
C     A, B, C
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 A(NXHP,NY,NZ),B(NXHP,NY,NZ),C(NXHP,NY,NZ)
C
C     D, E, F
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS:
C        COMPLEX*16 D(NXHP,NY,NZ),E(NXHP,NY,NZ),F(NXHP,NY,NZ)
C
      IMPLICIT REAL*8(A-H,O-Z)                                          
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      COMMON/WAV/ XW(129),YW(129),ZW(129),XSQ(129),YSQ(129),ZSQ(129)    
      COMPLEX*16  A(NXHP,NY,1),B(NXHP,NY,1),C(NXHP,NY,1),               
     +            D(NXHP,NY,1),E(NXHP,NY,1),F(NXHP,NY,1)                
      COMPLEX*16  IUNIT                                                 
      IUNIT = CMPLX(0.D0,1.D0)                                          
      DO 1 K=1,NZ                                                       
         RW= ZW(K)                                                      
         DO 1 J=1,NY                                                    
            QW= YW(J)                                                   
            DO 2 I=1,NXHP                                               
               A(I,J,K) = 0.25D0*(D(I,J,K) + 2.D0*A(I,J,K))             
               B(I,J,K) = 0.25D0*(E(I,J,K) + 2.D0*B(I,J,K))             
               C(I,J,K) = 0.25D0*(F(I,J,K) + 2.D0*C(I,J,K))             
               D(I,J,K) = (QW*C(I,J,K) - RW*B(I,J,K))*IUNIT             
               E(I,J,K) = (RW*A(I,J,K) - XW(I)*C(I,J,K))*IUNIT          
               F(I,J,K) = (XW(I)*B(I,J,K) - QW*A(I,J,K))*IUNIT          
    2       CONTINUE                                                    
    1 CONTINUE                                                          
      RETURN                                                            
      END
C
C                                                                       
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                       
      SUBROUTINE  LIN(AM,BM,CM,D,E,F,DT2,DTV,DTT)                       
C                                                                       
C     ONE TIME STEP OF THE LEAP-FROG SCHEME                             
C
C     ON ENTRY
C 
C     AM, BM, CM
C        COMPLEX*16 AM(NXHP,NY,NZ),BM(NXHP,NY,NZ),CM(NXHP,NY,NZ) 
C
C     D, E, F
C        COMPLEX*16 D(NXHP,NY,NZ),E(NXHP,NY,NZ),F(NXHP,NY,NZ) 
C
C     DTT, DT2
C        REAL*8, ARE DELTA TIME, AND DOUBLE DELTA TIME.
C
C     DTV
C        REAL*8, IS A VISCOSITY COEFFICIENT.
C
C
C     ON RETURN
C
C     AM, BM, CM
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS ABOVE. 
C
C     D, E, F
C        ARE THE THREE DIMENSIONAL ARRAYS SPECIFIED AS ABOVE. 
C
      IMPLICIT REAL*8(A-H,O-Z)                                          
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      COMMON/WAV/ XW(129),YW(129),ZW(129),XSQ(129),YSQ(129),ZSQ(129)    
      COMMON/TRUNC/XWT(129),YWT(129),ZWT(129)                           
      COMPLEX*16  AM(NXHP,NY,1),BM(NXHP,NY,1),CM(NXHP,NY,1),            
     +            D(NXHP,NY,1),E(NXHP,NY,1),F(NXHP,NY,1)                
      COMPLEX*16  P,IUNIT                                               
      IUNIT = CMPLX(0.D0,1.D0)                                          
      DO 1 K=1,NZ                                                       
         RW= ZW(K)                                                      
         RT= ZWT(K)                                                     
         ZWSQ = ZSQ(K)                                                  
         DO 2 J=1,NY                                                    
            QW= YW(J)                                                   
            QT= YWT(J)                                                  
            YWSQ = YSQ(J) + ZWSQ                                        
            DO 3 I=1,NXHP                                               
               FACT3 = XSQ(I) + YWSQ                                    
               FACT1 = DTV*DTT*FACT3                                    
               FACT2 = RT*QT*XWT(I)/(1.D0 + FACT1)                      
               D(I,J,K) = (1.D0 - FACT1)*AM(I,J,K) + DT2*D(I,J,K)       
               E(I,J,K) = (1.D0 - FACT1)*BM(I,J,K) + DT2*E(I,J,K)       
               F(I,J,K) = (1.D0 - FACT1)*CM(I,J,K) + DT2*F(I,J,K)       
               P= (F(I,J,K)*RW+E(I,J,K)*QW+D(I,J,K)*XW(I))/FACT3        
               AM(I,J,K)=(D(I,J,K)-XW(I)*P)*FACT2                       
               BM(I,J,K)=(E(I,J,K)-QW*P)*FACT2                          
               CM(I,J,K)=(F(I,J,K)-RW*P)*FACT2                          
               D(I,J,K) = (QW*CM(I,J,K) - RW*BM(I,J,K))*IUNIT           
               E(I,J,K) = (RW*AM(I,J,K) - XW(I)*CM(I,J,K))*IUNIT        
               F(I,J,K) = (XW(I)*BM(I,J,K) - QW*AM(I,J,K))*IUNIT        
    3       CONTINUE                                                    
    2    CONTINUE                                                       
    1 CONTINUE                                                          
      RETURN                                                            
      END
      SUBROUTINE VERIFYTR()
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
      OPEN(10, FILE='TURB3D.TEST.VERIFY')  
C.... THE VERIFICATION ROUTINE                                      
      WRITE(10,1)                                                       
    1 FORMAT(1X,'TURB3D BENCHMARK TEST VERIFICATION & TIMING'/)              
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
      IF((ABS(EU1(2)-0.624925082612E-01)/0.624925082612E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(2)-0.624925082612E-01)/0.624925082612E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(2)-0.156225003000E-07)/0.156225003000E-07).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(2)-0.624926254300E-01)/0.624926254300E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(2)-0.624926254300E-01)/0.624926254300E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(2)-0.249970001800E+00)/0.249970001800E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(3)-0.624849705524E-01)/0.624849705524E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(3)-0.624849705524E-01)/0.624849705524E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(3)-0.624651006823E-07)/0.624651006823E-07).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(3)-0.624854390410E-01)/0.624854390410E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(3)-0.624854390410E-01)/0.624854390410E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(3)-0.249939757280E+00)/0.249939757280E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(4)-0.624774493857E-01)/0.624774493857E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(4)-0.624774493857E-01)/0.624774493857E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(4)-0.140512172452E-06)/0.140512172452E-06).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(4)-0.624785032284E-01)/0.624785032284E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(4)-0.624785032284E-01)/0.624785032284E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(4)-0.249909516521E+00)/0.249909516521E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(5)-0.624698822596E-01)/0.624698822596E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(5)-0.624698822596E-01)/0.624698822596E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(5)-0.249719639172E-06)/0.249719639172E-06).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(5)-0.624717551627E-01)/0.624717551627E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(5)-0.624717551627E-01)/0.624717551627E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(5)-0.249879029611E+00)/0.249879029611E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(6)-0.624623316921E-01)/0.624623316921E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(6)-0.624623316921E-01)/0.624623316921E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(6)-0.390084991743E-06)/0.390084991743E-06).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(6)-0.624652573427E-01)/0.624652573427E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(6)-0.624652573427E-01)/0.624652573427E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(6)-0.249848546626E+00)/0.249848546626E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(7)-0.624547351769E-01)/0.624547351769E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(7)-0.624547351769E-01)/0.624547351769E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(7)-0.561551328264E-06)/0.561551328264E-06).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(7)-0.624589468415E-01)/0.624589468415E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(7)-0.624589468415E-01)/0.624589468415E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(7)-0.249817817667E+00)/0.249693943767E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(8)-0.624471552364E-01)/0.624471552364E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(8)-0.624471552364E-01)/0.624471552364E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(8)-0.764127911828E-06)/0.764127911828E-06).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(8)-0.624528862483E-01)/0.624528862483E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(8)-0.624528862483E-01)/0.624528862483E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(8)-0.249787092799E+00)/0.249787092799E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(9)-0.624395293607E-01)/0.624395293607E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(9)-0.624395293607E-01)/0.624395293607E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(9)-0.997746362297E-06)/0.997746362297E-06).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(9)-0.624470125518E-01)/0.624470125518E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(9)-0.624470125518E-01)/0.624470125518E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(9)-0.249756122144E+00)/0.249756122144E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(10)-0.624319045872E-01)/0.624319045872E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(10)-0.624319045872E-01)/0.624319045872E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(10)-0.126239922912E-05)/0.126239922912E-05).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(10)-0.624413727308E-01)/0.624413727308E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(10)-0.624413727308E-01)/0.624413727308E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(10)-0.249725093861E+00)/0.249725093861E+00).GE.1E-2)      
     +    IVALID=1                                                      
C
      IF((ABS(EU1(11)-0.624242648740E-01)/0.624242648740E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EV1(11)-0.624242648740E-01)/0.624242648740E-01).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EW1(11)-0.155808881559E-05)/0.155808881559E-05).GE.1E-2)       
     +    IVALID=1                                                      
      IF((ABS(EOX1(11)-0.624359507677E-01)/0.624359507677E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOY1(11)-0.624359507677E-01)/0.624359507677E-01).GE.1E-2)      
     +    IVALID=1                                                      
      IF((ABS(EOZ1(11)-0.249693943792E+00)/0.249693943792E+00).GE.1E-2)      
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

C                                                                       
      RETURN                                                              
C                                                                       
 4000 FORMAT (A,4x,I3,6x,2(1x,E12.4,' +/- ',E10.4))
C                                                                       
      END
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

