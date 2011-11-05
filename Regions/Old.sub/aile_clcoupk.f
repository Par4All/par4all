      PROGRAM AILECLCOUPK
      DIMENSION T(52,21,60)
      COMMON/CT/T
      COMMON/GAMMA/GAM,GAM1,GAM2,GAM3,GAM4,GAM5,GAM6,GAM7,GAM8,GAM9
      COMMON/CI/I1,I2,IMAX,I1P1,I1P2,I2M1,I2M2,IBF
      COMMON/CJ/J1,J2,JMAX,J1P1,J1P2,J2M1,J2M2,JA,JB,JAM1,JBP1
      COMMON/CK/K1,K2,KMAX,K1P1,K1P2,K2M1,K2M2
      COMMON/CNI/L
      DATA N1,N3,N4,N7,N10,N14,N17/1,3,4,7,10,14,17/

      READ(NXYZ) I1,I2,J1,JA,K1,K2,GAM

      GAM1=(GAM+1.)/(2.*GAM)
      GAM2=(GAM-1.)/(2.*GAM)
      GAM3=(GAM+1.)/(GAM-1.)
      GAM4=GAM-1.
      GAM5=1./(GAM-1.)
      GAM6=(GAM-1.)/GAM
      GAM7=GAM+1.
      GAM8=-GAM1*GAM2
      GAM9=-2./(GAM-1.)

      IF(J1.GE.1.AND.K1.GE.1) THEN
         N4=4
         J1=J1+1
         J2=2*JA+1
         JA=JA+1
         JAM1=JA-1
         K1=K1+1
         K2=K2+1

         I1P1=I1+1
         I1P2=I1+2
         I2M1=I2-1
         I2M2=I2-2
         
         NU =11
         LLU=21
         NW =LLU
         MU = 6

         CALL CLCOUPK(K1,J1,JAM1,NU,NW)
         CALL CLCOUPK(K1,J1,JAM1,NU,NW)

         DO II=I1P1,I2M1
            LLU=LU
            LU =NU
            NU =MU
            MU =MMU
            MMU=NW
            NW =LLU
         ENDDO

         DO K=K1,K2
            KK=K
            CALL CLCOUPK(KK,J1,JA,MU,NW)
         ENDDO
      ENDIF
      END

      SUBROUTINE CLCOUPK(K,JDEB,JFIN,NU,NW)
C*****************************************************
C     CONDITION DE CONTINUITE AUX POINTS DOUBLES
C     JH,KH,....HOMOLOGUES DE J,K,....
C*****************************************************
      DIMENSION T(52,21,60)
      COMMON/CT/T
      COMMON/GAMMA/GAM,GAM1,GAM2,GAM3,GAM4,GAM5,GAM6,GAM7,GAM8,GAM9
      COMMON/CJ/J1,J2,JMAX,J1P1,J1P2,J2M1,J2M2,JA,JB,JAM1,JBP1

      PX=0.
      PY=0.
      PZ=1.
C
      DO 600 J=JDEB,JFIN
      JH=J1+J2-J
C
      ROS =T(J,K,NW  )
      ROUS=T(J,K,NW+1)
      ROVS=T(J,K,NW+2)
      ROWS=T(J,K,NW+3)

      ROH =T(JH,K,NW  )
      ROUH=T(JH,K,NW+1)
      ROVH=T(JH,K,NW+2)
      ROWH=T(JH,K,NW+3)

      VOL=1./T(J,K,NU)
      VX=T(J,K,NU+1)*VOL
      VY=T(J,K,NU+2)*VOL
      VZ=T(J,K,NU+3)*VOL

      C0=GAM1-GAM2*(VX*VX+VY*VY+VZ*VZ)
      GAMVN=GAM2*(VX*PX+VY*PY+VZ*PZ)
      C1=SQRT(C0+GAMVN*GAMVN)
      VALPP=GAM3*GAMVN+C1
      VALPN=GAM3*GAMVN-C1

      IF(VALPP.LT.0.) THEN

C     FRONTIERE AVAL SUPERSONIQUE

 100  CONTINUE
      RO =ROS
      ROU=ROUS
      ROV=ROVS
      ROW=ROWS

C     FRONTIERE AVAL SUBSONIQUE

      ELSEIF(GAMVN.LT.0.) THEN
 200  CONTINUE
      VL=C1-GAMVN
      BX=VX+VL*PX
      BY=VY+VL*PY
      BZ=VZ+VL*PZ
      AX=ROUS-ROS*BX
      AY=ROVS-ROS*BY
      AZ=ROWS-ROS*BZ
      VLP=-C1-GAMVN
      DX=GAM6*VX+VLP*PX
      DY=GAM6*VY+VLP*PY
      DZ=GAM6*VZ+VLP*PZ
      D0=C0+VX*DX+VY*DY+VZ*DZ
      DD=D0*ROH-DX*ROUH-DY*ROVH-DZ*ROWH
      DA=DX*AX+DY*AY+DZ*AZ
      DB=DX*BX+DY*BY+DZ*BZ
      RO=(DA+DD)/(D0-DB)
      ROU=AX+RO*BX
      ROV=AY+RO*BY
      ROW=AZ+RO*BZ
      ELSEIF(VALPN.LT.0.) THEN

C     FRONTIERE AMONT SUBSONIQUE

 300  CONTINUE
      VL=-C1-GAMVN
      BX=VX+VL*PX
      BY=VY+VL*PY
      BZ=VZ+VL*PZ
      AX=ROUH-ROH*BX
      AY=ROVH-ROH*BY
      AZ=ROWH-ROH*BZ
      VLN=C1-GAMVN
      DX=GAM6*VX+VLN*PX
      DY=GAM6*VY+VLN*PY
      DZ=GAM6*VZ+VLN*PZ
      D0=C0+VX*DX+VY*DY+VZ*DZ
      DD=D0*ROS-DX*ROUS-DY*ROVS-DZ*ROWS
      DA=DX*AX+DY*AY+DZ*AZ
      DB=DX*BX+DY*BY+DZ*BZ
      RO=(DA+DD)/(D0-DB)
      ROU=AX+RO*BX
      ROV=AY+RO*BY
      ROW=AZ+RO*BZ
      ELSE

C     FRONTIERE AMONT SUPERSONIQUE

 400  CONTINUE
      RO =ROH
      ROU=ROUH
      ROV=ROVH
      ROW=ROWH

      ENDIF
 500  CONTINUE
      T(J,K,NW  )=RO
      T(J,K,NW+1)=ROU
      T(J,K,NW+2)=ROV
      T(J,K,NW+3)=ROW

      T(JH,K,NW  )=RO
      T(JH,K,NW+1)=ROU
      T(JH,K,NW+2)=ROV
      T(JH,K,NW+3)=ROW
C
  600 CONTINUE
      RETURN
      END
