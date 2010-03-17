      program save04

C     Check that offsets in the static area are correct regardless of the
C     declaration order

      save x
      save y

      doubleprecision x

      common /foo/z, t

      save u

      real z(3), y(6)

      x = y + w

      end
