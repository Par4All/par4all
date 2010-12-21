      program HybridComp02
      integer i
      integer j
      integer n
      parameter (n=5)
      integer a(n, n)
      do 10 i = 1, n
         do j = 1, n
            a(i,j) = i*j
         enddo
 10   continue
      PRINT *,a
      end
