C     Attempt at reducing the context of a bug found in
C     ValidationPrivate/advbnd.f.

C     Because of the variable sort, nyy is used before it is defined
C     which is forbidden by FOrtran 77 standard, at least according to
C     SUN compiler. If NCL is defined before NYY, this module is not
C     legal Fortran 77 code.

      subroutine entry22

c      INTEGER*4 NCL
c      PARAMETER (NCL = 3*NYY)

      INTEGER*4 NYY
      PARAMETER (NYY = 514)

      INTEGER*4 NCL
      PARAMETER (NCL = 3*NYY)

      common /foo/ x(ncl)

      print *, x

      entry inibnd

      x(1) = 0.

      end
