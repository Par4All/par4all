! Check the computation of a lower bound for a min and of an upper bound
! for a max when all the arguments are themselves bounded

      subroutine minmax4

      if(4.gt.m.or.m.gt.10) stop
      if(5.gt.n.or.n.gt.11) stop

      i = min(m, n)
      j = max(m, n)

      print *, i, j

      end
