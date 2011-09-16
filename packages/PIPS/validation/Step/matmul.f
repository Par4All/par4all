      PROGRAM MATMUL
      implicit none
      INTEGER  N, I, J, K
      PARAMETER (N=10)
      REAL*8 A(N,N), B(N,N), C(N,N)

!$OMP PARALLEL

C     Initialize A, B and C matrices
!$OMP DO
      DO 10 J=1, N
        DO 10 I=1, N
          A(I,J) = 2.0
          B(I,J) = 3.0
          C(I,J) = 0
  10  CONTINUE
!$OMP END DO

!$OMP MASTER
      PRINT *, '******************************************************'
      PRINT *, 'Matrices initialized'
      PRINT *, '******************************************************'
!$OMP END MASTER

C     Compute matrix-matrix product
!$OMP DO
      DO 20 J=1, N
        DO 20 I=1, N
          DO 20 K=1, N
            C(I,J) = C(I,J) + A(I,K) * B(K,J)
  20  CONTINUE
!$OMP END DO

C     Print result matrix
!$OMP MASTER
      PRINT *, '******************************************************'
      PRINT *, 'Result Matrix:'
      DO 50 I=1, N
        DO 40 J=1, N
          WRITE(*,30) C(I,J)
  30	  FORMAT(2x,f8.2,$)
  40      CONTINUE
          PRINT *, ' '
  50      CONTINUE
      PRINT *, '******************************************************'
      PRINT *, 'Done.'
!$OMP END MASTER
!$OMP END PARALLEL

      END

