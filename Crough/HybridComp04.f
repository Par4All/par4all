      program HybridComp04
      integer*4 i
      integer*4 j
      integer*4 n
      parameter (n=5)
      integer*4 a(n, n)
      do 10 i = 1, n
         do j = 1, n
            a(i,j) = i*j
         enddo
 10   continue
      PRINT *,a
      end
