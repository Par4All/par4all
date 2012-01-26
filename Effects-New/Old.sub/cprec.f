      program cprec

C     check that precondition holding on common variables
C     are passed down as they should

C     check that regions are passed upwards correctly

      common /composi/ jpoint
      common /compos/ eff(401)

      do j = 1, n
         jpoint = j
         call vs()
      enddo

      end
      subroutine vs
      common /composi/ jpoint
      common /compos/ eff(401)

      eff(jpoint) = 0.0

      end
