      subroutine minmax(i,j)
      common /foo/ k, l

      i = min0(k, l+2)

      j = max0(k, l*l, 4)

      end
