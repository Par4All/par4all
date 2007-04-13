C     I is indirectly modified by many procedures via a hidden common

      program hidden

      call init

      call incr(1)

      call double

      call printi

      end

      subroutine init
      common /foo/i

      i = 0

      end

      subroutine incr(j)
      common /foo/i

      i = i + j

      end

      subroutine double
      common /foo/i

      i = 2 * i

      end

      subroutine printi
      common /foo/i

      print *, i

      end
