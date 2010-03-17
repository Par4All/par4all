      program matini
      integer size
      parameter (size=100)
      real t(size,size)
      
      do 100 i = 1, size
         do 200 j = 1, size
            t(i,j) = 0.
 200     continue
 100  continue
      
      end
