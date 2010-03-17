      SUBROUTINE EARLY(N)

      M=10
      DO 10 I = 1,M
         DO 20 J = 1,N,M
            T = T + 1.0
 20      CONTINUE
 10   CONTINUE
      M=20
      DO 30 I = 1,M
         DO 40 J = 1,N,M
            T = T + 1.0
 40      CONTINUE
 30   CONTINUE
      END
