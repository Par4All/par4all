      subroutine cleandec

      common /foo/i,j,c(2), m, n
      common /ufoo/a,b,d
      integer o,p,q
      parameter (k=10)
      parameter (w=5)	
      equivalence (o,p)
      if(i.gt.j) then
      call bar
      endif
      print *, i, j, m, m, p

      end

      subroutine bar
      common /foo/i,j,k,l,m,n
      common /ufoo/a,b,d
      parameter (x=10)
      parameter (y=5*x)	
      if(i.gt.j) then
         i = i + 1
         j = j + 2 * m
         k = k + 3
         l = l + 4 *y
      endif

      end
