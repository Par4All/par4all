      subroutine type23(s)

C     Variable and constant lengths must be taken into account and
C     substring applied... except when you cannot and have to assume
C     it's right

      character*(*) s

C     Should not be truncated since the actual lenght of s is unknown
      s = "Hello World!"

      end
