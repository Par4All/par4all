      subroutine equiv23

C     Nga Nguyen: Fortran-77 extension for adjustable arrays. PIPS
C     should not core dump but indicate that the equivalence statement
C     is not acceptable.

      common m,n

      real a(m:n), b(m:n)

      equivalence (a(3), b(2))

      end
