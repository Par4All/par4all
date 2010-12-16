      program HybridComp01
      integer i
      integer j
      integer a(5, 5)
      do 10 i = 1, 5
         do j = 1, 5
            a(i,j) = i*j
         enddo
 10   continue
      PRINT *,a
      end
