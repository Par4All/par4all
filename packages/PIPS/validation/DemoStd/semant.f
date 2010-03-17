      subroutine semant(a, n)
      parameter (maxsiz = 10)
      data m /6/
      real a(n)
      equivalence (i, j)

C     You cannot be sure that M==6 because there may be several calls to SEMANT

      if(n.ge.maxsiz) then
         n = maxsiz
      endif

C     This loop can be parallelized

      do 1 i = 1, n, 2
         a(i) = 0.
         a(j+1) = 0.
 1    continue

C     M is a static variable because of the DATA statement
C     Its value is inherited from one call to the next and
C     pretty much unknown. Although it looks as if it were
C     uninitialized, it is at least initialized by DATA.

      call swap(n, m)
      print *, n, m
      end
      subroutine swap(i,j)
      itmp = i
      i = j
      j = itmp
      return
      end
