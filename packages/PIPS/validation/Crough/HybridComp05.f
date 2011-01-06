      program HybridComp05
      integer*8 i
      integer*8 j
      integer*8 n
      parameter (n=5)
      integer*8 a(n, n)
      do 10 i = 1, n
         do j = 1, n
            a(i,j) = i*j
         enddo
 10   continue
      PRINT *,a
      end
