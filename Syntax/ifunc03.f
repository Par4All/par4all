      PROGRAM IFUNC03
      CALL BLA()
      END

      SUBROUTINE BLA()
C     This call should not be handled by transformer analysis
C     because o the implicit type conversion
      I = FOO()
      END

      FUNCTION FOO()
      FOO = 0
      END

