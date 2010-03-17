      program matmul
      INTEGER SIZE
      PARAMETER (SIZE=100)
      real a(size,size), b(size,size), c(size,size)

      do 100 i = 1, size
         do 100 j = 1, size
            do 100 k = 1, size
               a(i,j) = a(i,j) + b(i,k)*c(k,j)
 100  continue

      end
