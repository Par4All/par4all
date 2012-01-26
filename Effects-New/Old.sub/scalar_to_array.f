      program PROG
      real A(100), B(2:5,6:8)
      call TOTO(A)
      call TOTO(B)
      call TOTO(B(3,7))
      end

      subroutine TOTO(S)
      real S

      S=0.
      end
