      PROGRAM NGA07
C
  
      REAL WLOOP(13,13)

C     To understand results for nga01 better: what do transformers show?

      read *, n, m

      do i = k, l
         do j = 1, m
            wloop(i,j) = 0.
         enddo
      enddo

      END
      
