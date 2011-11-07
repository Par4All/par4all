C
      SUBROUTINE MM(N, A, B, C)
      INTEGER N
      REAL A(N, *), B(N, *), C(N, *)
C
      DO I = 1, N
         CALL SMXPY(N, A(1, I), C(1, I), B)
      ENDDO
C
      RETURN
      END
C
C
C
      SUBROUTINE SMXPY(D, Y, X, M)
      INTEGER D
      REAL X(*), Y(*), M(D, *)
C
      DO J = 1, D
         Y(J) = 0
         DO K  = 1, D
            Y(J) = Y(J) + X(K)*M(J, K)
         ENDDO
      ENDDO
C
      RETURN
      END
