      subroutine large01

C     Check the behavior of large integer constants stored on 32 or 64 bits
C     in declarations.

      real x(1234567890)
      print *, x(1)

      end

      subroutine large02

      real x(12345678901)
      print *, x91)

      end

      subroutine large03

      real x(12345678901234567890)
      print *, x(1)

      end

      subroutine large04

      real x(123456789012345678901234567890)
      print *, x(1)

      end

      subroutine large05

      real x(1234567890123456789012345678901234567890)
      print *, x(1)

      end

      subroutine large06

      real x(12345678901234567890123456789012345678901234567890)

      print *, x(1)

      end
