! To check impact of non-affine expressions as arguments of min and max

      subroutine minmax5

      if(4.gt.m.or.m.gt.10) stop
      if(5.gt.n.or.n.gt.11) stop

      i = min(m/2, n/2)
      j = max(m/2, n/2)

      print *, i, j

      end

