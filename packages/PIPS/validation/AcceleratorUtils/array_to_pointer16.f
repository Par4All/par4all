!!
!! file for P4A_KERNEL_0.f
!!
      SUBROUTINE P4A_KERNEL_0(N, A, B, I, J)
      INTEGER N, I, J
      INTEGER A(1:N, 1:N), B(1:N, 1:N)
C Loop nest P4A end
      IF (J.LE.N-1.AND.I.LE.N-1) THEN
         A(I+1,J+1) = 1
         B(I+1,J+1) = 1
      ENDIF
      END
