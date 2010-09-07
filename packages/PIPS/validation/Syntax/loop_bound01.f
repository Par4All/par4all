!     Check replacement of non-affine loop bounds

!     Bug found in BU_INTER_su2cor.f, loop 200, subroutine LOOPS

!     The bug is reproduced with property PARSER_LINEARIZE_LOOP_BOUNDS

      program loop_bound01

      real a(16)

      n = 4
!     U_0 is going to be used
      do i = 1, 2**n
         a(i) = 0.
      enddo

!     Oops, U_0 is going to be re-used
      do i = 1, 2**n
         a(i) = float(i)
      enddo

      end
