C     Attempt at reducing the context of a bug found in
C     ValidationPrivate/advbnd.f.

C     Because of the variable sort, nyy is used before it is defined

      subroutine entry23

      REAL*4 NCL
      PARAMETER (NCL = 3.05)

      common /foo/ x(ncl)

      print *, x

      entry inibnd

      x(1) = 0.

      end
