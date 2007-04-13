      program edited05

C     The parser must be able to process the same routine several times
C     after user edition steps, but not again when it is no longer
C     useful

      call bar(y)

      print *, ibar(z)

      end

      subroutine bar(y)

!!      integer x(3)
!%      real x(10)

      print *, x

      end
