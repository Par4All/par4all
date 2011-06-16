      SUBROUTINE A92(N, A, B, C, Y, Z)
      INTEGER N
      REAL A(*), B(*), C(*), Y(*), Z(*)
      INTEGER I

      DO I=1,N
            C(I) = (A(I) + B(I)) / 2.0
      ENDDO

      DO I=1,N
            Z(I) = SQRT(C(I))
      ENDDO

      DO I=2,N+1
            Y(I) = Z(I-1) + A(I)
      ENDDO
      END
