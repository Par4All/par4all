      program equiv01

C     Check that the error in EQUIVALENCE is spotted by PIPS: array T would
C     have a negative offset in common /C_U/

      real t(10), u(10)
      common /c_u/u
      equivalence (t(2),u(1))

      do i = 1, 10
         t(i) = u(i)
      enddo

      end
