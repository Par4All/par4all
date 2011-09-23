      PROGRAM NORMMAIN
      DIMENSION T(52,21,60)
      COMMON/CT/T
      COMMON/CI/I1,I2,IMAX,I1P1,I1P2,I2M1,I2M2,IBF
      COMMON/CJ/J1,J2,JMAX,J1P1,J1P2,J2M1,J2M2,JA,JB,JAM1,JBP1
      COMMON/CK/K1,K2,KMAX,K1P1,K1P2,K2M1,K2M2
      COMMON/CNI/L
      DATA N1,N3,N4,N7,N10,N14,N17/1,3,4,7,10,14,17/

      READ(NXYZ) I1,I2,J1,JA,K1,K2
      REWIND NXYZ
C     
      IF(J1.GE.1.AND.K1.GE.1) THEN
         N4=4
         J1=J1+1
         J2=2*JA+1
         JA=JA+1
         K1=K1+1
         CALL NORM(N10,N7,N4,N14,N17,I2)
      ENDIF
      END

      SUBROUTINE NORM(LI,NI,MI,NN,NC,I) 
C***************************************************************
C     CALCULE LES NORMALES
C     LI,NI,MI  : PLANS (I+1), I ,(I-1)
C     NN : RESULTATS (UNIQUEMENT POUR LES NORM. AUX PLAN I1 ET I2)
C     NC : STOCKAGE DES NORMALES DANS LE CADRE DU TABLEAU T
C     MAILLAGE SYMETRIQUE / XOY
C     CALCULS SPECIFIQUES A L AILE ETUDIEE
C****************************************************************
      DIMENSION T(52,21,60)
      DIMENSION TI(3)

      COMMON/T/T
      COMMON/I/I1,I2,IMAX,I1P1,I1P2,I2M1,I2M2,IBF
      COMMON/J/J1,J2,JMAX,J1P1,J1P2,J2M1,J2M2,JA,JB,JAM1,JBP1
      COMMON/K/K1,K2,KMAX,K1P1,K1P2,K2M1,K2M2
      COMMON/IO/LEC,IMP,KIMP,NXYZ,NGEO,NDIST

C ....

      DO 300 K=K1,K2
      DO 300 J=J1,JA

      CALL PVNMUT(TI)
      T(J,K,NN  )=S*TI(1)
      T(J,K,NN+1)=S*TI(2)
      T(J,K,NN+2)=S*TI(3)
  300 CONTINUE

C ....

      END

      SUBROUTINE PVNMUT(C)
C********************************************
C     ECRITURE DE C
C********************************************
      DIMENSION C(3), CX(3)
      CX(1)= 1
      CX(2)= 2
      CX(3)= 3
      R=SQRT(CX(1)*CX(1)+CX(2)*CX(2)+CX(3)*CX(3))
      IF(R.LT.1.E-12) R=1.
      DO I = 1,3
      C(I) = CX(I)/R
      ENDDO
      RETURN
      END
