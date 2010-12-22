      program HybridComp03
      integer i
      integer j
      integer n
      integer a
      n = 5
      a = 0
      do 10 i = 1, n
         do j = 1, n
            a = i*j
         enddo
 10   continue
      PRINT *,a
      end
