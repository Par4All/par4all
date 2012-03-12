      program HybridComp15
      integer i
      integer j
      real a(5, 5)
      real cst(5)
      cst (1) = 1.0
      cst (2) = 2.0
      cst (3) = 3.0
      cst (4) = 4.0
      cst (5) = 5.0
      do 10 i = 1, 5
         do j = 1, 5
            a(i,j) = cst(i) * 2.4E5
         enddo
 10   continue
      PRINT *,a
      end
