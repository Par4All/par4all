      program add
      INTEGER SIZE
      PARAMETER (SIZE=100)
      real a(size,size), b(size,size), c(size,size)
      
      
      do 100 i = 1, size
         do 200 j = 1, size
            b(i,j) = (i-1)*size+(j-1)
            c(i,j) = b(i,j)/(size*size)
 200     continue
 100  continue
      
      do 300 i = 1, size
         do 400 j = 1, size
            a(i,j) = b(i,j) + c(i,j)
 400     continue
 300  continue
      
      end
