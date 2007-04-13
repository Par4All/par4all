      program inter03

C     Check that equivalences are taken into account properly

      equivalence (x, j)

      call incr(j)

      print *, j

      end

      subroutine incr(i)
      i = i + 1
      end

