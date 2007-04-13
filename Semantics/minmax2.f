      subroutine minmax2(m,n,i,j,k,l)

      i = min(m,n)
      j = min0(m,n)

      k = max(m,n)
      l = max0(m,n)

      print *, i, j, k, l

      i = min(i, j, k, l, m, n)

      print *, i

      end

