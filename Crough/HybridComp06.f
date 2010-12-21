      program HybridComp06
      integer*8 i
      integer*8 j
      integer*8 n
      integer*8 a
      parameter (n = 5)
      a = 0
      do 10 i = 1, n
         do j = 1, n
            a = i*j
         enddo
 10   continue
      PRINT *,a
      end
