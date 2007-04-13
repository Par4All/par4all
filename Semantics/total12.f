      SUBROUTINE TOTAL12

C     Submitted by Nga Nguyen, excerpt from INITIA in SWIM.f (Perfect
C     Club)

      PARAMETER (N1 = 513, N2 = 513)
      COMMON UNEW(N1,N2)
      COMMON /CONS/ M,N
      READ (5,*) M, N

      DO 20 I = 1, M
         DO 20 J = 1, N
            UNEW(I+1,J) = (I+1)*J
 20   CONTINUE

      DO 30 J = 1, N
         UNEW(1,J) = UNEW(M+1,J)
 30   CONTINUE

      DO 40 I = 1, M
         UNEW(I+1,N+1) = UNEW(I+1,1)
 40   CONTINUE

      UNEW(1,N+1) = UNEW(M+1,1)
      END
