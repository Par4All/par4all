C   EXAMPLE Program
C
      SUBROUTINE TRY
      PARAMETER (NJA=128, NJB=128, JL=1, JU=128, KL=1, KU=128)
      COMMON /ARRAYS/ A(NJA,NJB), B(NJA,NJB), C(NJA,NJB), D(NJA,NJB),
     $ E(NJA,NJB), F(NJA,NJB,3), X(NJA,NJB), Y(NJA,NJB), FX(NJA,NJB,3)
C
      J = JL
      DO 1 K = KL,KU
        RLD = C(J,K)
        RLDI = 1./RLD
        F(J,K,1) = F(J,K,1)*RLDI
        F(J,K,2) = F(J,K,2)*RLDI
        F(J,K,3) = F(J,K,3)*RLDI
        X(J,K) = D(J,K)*RLDI
        Y(J,K) = E(J,K)*RLDI
1     CONTINUE
C
      J = JL+1
      DO 2 K = KL,KU
        RLD1 = B(J,K)
        RLD = C(J,K) - RLD1*X(J-1,K)
        RLDI = 1./RLD
        F(J,K,1) = (F(J,K,1) - RLD1*F(J-1,K,1))*RLDI
        F(J,K,2) = (F(J,K,2) - RLD1*F(J-1,K,2))*RLDI
        F(J,K,3) = (F(J,K,3) - RLD1*F(J-1,K,3))*RLDI
        X(J,K) = (D(J,K) - RLD1*Y(J-1,K))*RLDI
        Y(J,K) = E(J,K)*RLDI
2     CONTINUE
C
      DO 3 J = JL+2,JU-2
        DO 11 K = KL,KU
          RLD2 = A(J,K)
          RLD1 = B(J,K) - RLD2*X(J-2,K)
          RLD = C(J,K) - (RLD2*Y(J-2,K) + RLD1*X(J-1,K))
          RLDI = 1./RLD
          F(J,K,1) = F(J,K,1) - F(J-2,K,1)
          F(J,K,2) = F(J,K,2) - F(J-2,K+1,2)
11      CONTINUE
3     CONTINUE
C
	K9=9
	K8 = K9 -1
	PRINT *, K8
	K9=80
      RETURN
      END
