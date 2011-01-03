      PROGRAM EXTRMAIN
      DIMENSION T(52,21,60)

      READ(NXYZ) I1,I2,J1,JA,K1,K2
      REWIND NXYZ
C
      IF(J1.GE.1.AND.K1.GE.1) THEN
         N4=4
         J1=J1+1
         J2=2*JA+1
         JA=JA+1
         K1=K1+1
         CALL EXTR(NI,NC,T,J1,JA, K1)
      ENDIF
      END
      SUBROUTINE EXTR(NI,NC,T,J1,JA, K1)
C********************************************************
C     CALCULE LES COEFFICIENTS D EXTRAPOLATION
C     SUR LA SURFACE DE L AILE  (K=K1)
C********************************************************
      DIMENSION T(52,21,60)

      L=NI
      K=K1
      DO 300 J=J1,JA
         S1=D(J,K  ,J,K+1,T,0)
         S2=D(J,K+1,J,K+2,T,0)+S1
         S3=D(J,K+2,J,K+3,T,0)+S2
         T(J,1,NC+3)=S2*S3/((S1-S2)*(S1-S3))
         T(J,1,NC+4)=S3*S1/((S2-S3)*(S2-S1))
         T(J,1,NC+5)=S1*S2/((S3-S1)*(S3-S2))
         JH=J1+J2-J
         T(JH,1,NC+3)=T(J,1,NC+3)
         T(JH,1,NC+4)=T(J,1,NC+4)
         T(JH,1,NC+5)=T(J,1,NC+5)
 300  CONTINUE

      END
      REAL FUNCTION D(J,K,JP,KP,T,L)
C*****************************************
C     CALCULE D=DISTANCE
C*****************************************
      DIMENSION T(52,21,60)
C
      D=SQRT((T(J,K,L  )-T(JP,KP,L  ))**2
     1     +(T(J,K,L+1)-T(JP,KP,L+1))**2
     2     +(T(J,K,L+2)-T(JP,KP,L+2))**2)
      END
