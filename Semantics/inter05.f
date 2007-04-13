      program inter05

C     Check that equivalences are taken into account properly

      integer k(10)

      equivalence (k, j)

      call incr(j)

      print *, j

      end

      subroutine incr(i)
      i = i + 1
      end

