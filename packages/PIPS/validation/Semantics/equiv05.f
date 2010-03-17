      program equiv05

C     Check static alias use for non-type compatible variables

      equivalence (x, i)

      x = 4.

      i = 5

      print *, 'x=', x, 'i=', i

      end
