
                                                                       
      SUBROUTINE  ZFFT(JN,NTSK,IND,IS)                                  
      IMPLICIT    REAL*8(A-H,O-Z)                                       
      PARAMETER   (IX= 64,IY= 64,IZ= 64 ,M1=6)

      PARAMETER   (IXPP=IX+2,IXHP=IX/2+1,ITOT=IXPP*IY*IZ,               
     +          IXY=IXHP*IY,IXZ=IXHP*IZ,IYZ=IY*IZ)                      
      COMMON/ALL/ U(IXPP,IY,IZ)        
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      REAL*8      AUX1(4*IX),AUX2(2*IX)                                  
      CALL DCFT(M1,U(1,JN,1),NXY,1,U(1,JN,1),NXY,1,NZ, NXHP,1,SZ,  
     +     AUX1,AUX2,JN)                                                
      RETURN                                                            
      END
C     
      SUBROUTINE  XYFFT(KN,NTSK,IND,IS)                                 

      IMPLICIT    REAL*8(A-H,O-Z)                                       
      PARAMETER   (IX= 64,IY= 64,IZ= 64 ,M1=6)

      PARAMETER   (IXPP=IX+2,IXHP=IX/2+1,ITOT=IXPP*IY*IZ,               
     +          IXY=IXHP*IY,IXZ=IXHP*IZ,IYZ=IY*IZ)                      
      COMMON/ALL/ U(IXPP,IY,IZ)       
      COMMON /DIM/ NX,NY,NZ,NXPP,NXHP,NTOT,NXY,NXZ,NYZ                  
      REAL*8      AUX1(4*IX),AUX2(2*IX)                            
      CALL DCFT(M1,U(1,1,KN),NXHP,1,U(1,1,KN),NXHP,1,NY,NXHP,1,SY,
     +     AUX1,AUX2,KN)                                               
      RETURN                                                            
      END
                                                                  
      SUBROUTINE DCFT( M1, X, INC1X, INC2X, Y, INC1Y, INC2Y, N, M,
     &   ISIGN, SCALE, RA, AUX,NQ)

      IMPLICIT REAL*8(A-H,O-Z)
      REAL*8 X(*),Y(10),RA(4*N),AUX(2*N)
C          CALL CFFT(0,M1,AUX,RA,RA(2*N+1),NQ)
        DO 10 I=1,M
          IBR = (I-1)*INC2X*2 + 1
          IBC = (I-1)*INC2X*2 + 2
          DO 20 II=1,N
            INDR = IBR + (II-1)*2*INC1X
            INDC = IBC + (II-1)*2*INC1X
            RA(II) = X(INDR)
            RA(II+N) = X(INDC)
20        CONTINUE
C          CALL CFFT(ISIGN,M1,AUX,RA,RA(2*N+1),NQ)
          DO 30 II=1,N
            INDR = IBR + (II-1)*2*INC1X
            INDC = IBC + (II-1)*2*INC1X
             Y(INDR) = SCALE*RA(II)
             Y(INDC) = SCALE*RA(II+N)
30         CONTINUE
10     CONTINUE
      RETURN
      END













