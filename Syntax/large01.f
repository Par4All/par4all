      subroutine large01

C     Check the behavior of large integer constants stored on 32 or 64 bits

      i = 1234567890
      print *, i

      end

      subroutine large02

      j = 12345678901
      print *, j

      end

      subroutine large03

      k = 12345678901234567890
      print *, k

      end

      subroutine large04

      l = 123456789012345678901234567890
      print *, l

      end

      subroutine large05

      m = 1234567890123456789012345678901234567890
      print *, m

      end

      subroutine large06

      n = 12345678901234567890123456789012345678901234567890
      print *, n

      end
