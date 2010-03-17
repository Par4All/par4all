      PROGRAM MG3XDEMO 
      INTEGER LM, NM, NV, NR, NIT
      PARAMETER( LM=6 )
      PARAMETER( NM=2+2**LM, NV=NM**3 )
      PARAMETER( NR = (8*(NM**3+NM**2+5*NM-23+7*LM))/7 )
      COMMON /X/ U, V, R, A, C, IR, MM
      REAL*8 U(NR),V(NV),R(NR),A(0:3),C(0:3)
      REAL*8 RNM2, RNMU, OLD2, OLDU
      REAL*8 XX
      INTEGER IR(LM), MM(LM)
      INTEGER IT, N
      DO 20 IT=1,NIT
        CALL MG3P(U,V,R,N,A,C,NV,NR,IR,MM,LMI)
 20   CONTINUE
      END
      SUBROUTINE MG3P(U,V,R,N,A,C,NV,NR,IR,MM,LM)
      INTEGER N,NV,NR,LM
      REAL*8 U(NR),V(NV),R(NR)
      REAL*8 A(0:3),C(0:3)
      INTEGER IR(LM), MM(LM)
      INTEGER K, J
      K = 1
      CALL PSINV(R(IR(K)),U(IR(K)),MM(K),C)
      IF( LM .EQ. 2 )GO TO 200
      DO 100 K = 2, LM-1
      J = K-1
      CALL PSINV(R(IR(K)),U(IR(K)),MM(K),C)
 100  CONTINUE
 200  CONTINUE
      J = LM - 1
      K = LM
      CALL PSINV(R,U,N,C)
      END
      SUBROUTINE PSINV(R,U,N,C)
      INTEGER N
      REAL*8 U(N,N,N),R(N,N,N),C(0:3)
      INTEGER I3, I2, I1
      CALL COMM3(U,N)
      END
      SUBROUTINE COMM3(U,N)
      INTEGER N
      REAL*8 U(N,N,N)
      INTEGER I3, I2, I1
      DO 100 I3=2,N-1
        DO 100 I2=2,N-1
          U(1,I2,I3) = U(N-1,I2,I3)
 100  CONTINUE
      END
