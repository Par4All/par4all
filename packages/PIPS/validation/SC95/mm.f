C     
C     MATRIX MULTIPLICATION - VERSION WITH CALL TO SAXPY
C     
C     DATA FLOW INTERPROCEDURAL ANALYSIS
C     SUMMARY OF EFFECTS ON A PROCEDURE
C     PARALLELIZATION OF LOOPS INCLUDING CALLS TO PROCEDURE
C     
      SUBROUTINE MM(N, A, B, C)
C     
      REAL*8 A(N,N), B(N,N), C(N,N), XAUX(0:127)
C     
      DO I = 1,N
         DO J = 1,N
            C(I,J) = 0.0
         ENDDO
      ENDDO
C     
      DO J = 1,N
         DO K = 1,N
            CALL SAXPY(N, C(1,J), A(1,K), B(K,J))
         ENDDO
      ENDDO
C
      RETURN
      END
C     
C     COMPUTATION OF X = X + S*Y
C     
C     SUMMARY OF EFFECTS ON A PROCEDURE
C     
      SUBROUTINE SAXPY(N, X, Y, S)
C     
      INTEGER N
      REAL*8 X(N), Y(N), S
C     
      DO 10 I = 1, N
         X(I) = X(I) + S*Y(I)
 10   CONTINUE
C
      END
