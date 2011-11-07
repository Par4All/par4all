C Nga Nguyen 5 March 2002 : this example violates the Fortran standard 
C Adjustable array's bounds must be a dummy argument or in a common block
C DIMENSION A(N,N),B(N,N),RL(N,N)

C***********************************************************************
C Excerpt from Cr2cnf from ONERA
C Purpose: in regions of Z2
C***********************************************************************
      SUBROUTINE NBDIAG(A,B,RL,NP)
      IMPLICIT REAL*8(A-H,O-Z)
C  ROUTINE DE CALCUL DE A=A-TB*DIAG(RL)*B .
      COMMON/TOTO/Z1(640),Z2(640),Z3(640)
      DIMENSION A(N,N),B(N,N),RL(N,N)
C     added to simulate actual preconditions
      N = 3*NP
      DO 1 K=1,N
1     Z1(K)=RL(K*(N+1)-N,1)
      DO 2 J=1,N
        DO 3 K=1,N
3       Z2(K)=Z1(K)*B(K,J)
        DO 4 K=1,N,3
        DO 4 I=1,N
 4      A(I,J)=A(I,J)-B(K,I)*Z2(K)-B(K+1,I)*Z2(K+1)-B(K+2,I)*Z2(K+2)
2     CONTINUE
      RETURN
      END
