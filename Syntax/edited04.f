      program edited04

C     The parser must be able to process the same routine several times
C     after user edition steps

      call bar(y)

      print *, ibar(z)

      end

      subroutine bar(y)

!!      integer x(3)
!%      real x(10)

      print *, x

      end

      function ibar(y)

!!      integer x(3)
!%      real x(10)

      print *, x

      ibar = 1

      end
