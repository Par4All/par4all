      program equiv12

C     Check that an external cannot appear in an EQUIVALENCE: PIPS does not even 
C     though it is declared so *before* the EQUIVALENCE statement

C     The external declaration is not checked by the parser using the
C     code_declarations list. Too bad, but PIPS is not supposed to cope with
C     non-Fortran compliant codes.

      external v
      real y(100)
      equivalence (x, z), (u, v)
      real u(100)

      do i = 2, 10
         y(i) = 0.
      enddo

      print *, i, j

      end
