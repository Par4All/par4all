      program HybridComp07
      integer i
      integer j
      integer*8 a(5, 5)
      do 10 i = 1, 5
         do j = 1, 5
            a(i,j) = i * 1147483648
         enddo
 10   continue
      PRINT *,a
      end
