      program HybridComp18
      integer i
      integer j
      integer n
      integer a (5,5)
      n = 5
      a = 0
      do 10 i = 1, n
         do 20 j = 1, n
            a(i,j) = i*j
 20      continue
 10   continue
      print *,a
      end
