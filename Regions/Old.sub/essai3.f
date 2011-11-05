C
C
      SUBROUTINE ESSAI3(N, A, B, X, Y)
      INTEGER N
      REAL A(N,N), B(N), X(N), Y(N)
C
      DO K = 1, 10
         DO I = 1,N
            DO J = 1,I
               Y(I) = Y(I) + A(I,J)
            ENDDO
         ENDDO
C
         CALL FUNC(N, X, A)
      ENDDO
C
      RETURN
      END
C
C
C
      SUBROUTINE FUNC(N, Y, A)
      INTEGER N
      REAL A(N, N), Y(N)
C
      DO I = 1,N
         DO J = I+1,N
            Y(I) = Y(I) + A(I,J)
         ENDDO
      ENDDO
C
      RETURN
      END
