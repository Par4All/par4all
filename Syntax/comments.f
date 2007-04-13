      SUBROUTINE COMMENTS
C
C     Different characters can be used to start a comment line:
C      C, * (both are defined by the ANSI standard), 
C      D (for debug; MIL-STD-1753?)
C      ! (for Fortran90)

C     This is an ANSI comment
      I = 1
*     This also is an ASCII comment
      I = 2
D     This is a debug statement: print *, i
      I = 3
!     This starts a Fortran 90 comment
      I = 4
X     This isn't a proper comment line
      I = 5
      END
