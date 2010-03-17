      program tticfg

      i = 2
      call foo (i)
      print *,i
      end

      subroutine foo (j)
      k = j + 1
      call bar (k)
      end

      subroutine bar (l)
      l = l + 2
      end

