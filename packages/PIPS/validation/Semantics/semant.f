      program callsemant
      real b(100)

      read *, l
      call semant(b,l)

      end

      subroutine semant(a, n)
      parameter (maxsiz = 10)
      data m /6/
      real a(n)
      equivalence (i, j)

      if(n.ge.maxsiz) then
         n = maxsiz
      endif

      do 1 i = 1, n, 2
         a(i) = 0.
         a(j+1) = 0.
 1    continue

      call swap(n, m)
      print *, m, n
      end
      subroutine swap(i,j)
      itmp = i
      i = j
      j = itmp
      return
      end
