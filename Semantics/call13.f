! Test for Ticket 561: the value of I is destroyed by the call to INC
! although the formal parameter is an array

      PROGRAM MAIN
      INTEGER I
      I = 2
      CALL INC(I)
      PRINT *, I
      END

      SUBROUTINE INC(A)
      INTEGER A(1)

      A(1) = A(1) + 1

      RETURN
      END
