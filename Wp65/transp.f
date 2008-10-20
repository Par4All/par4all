      program transp
      INTEGER SIZE
      PARAMETER (SIZE=100)
      real m(size,size)
     
      do 100 i = 1, size-1
         do 200 j = i+1, size
            t = m(i,j)
            m(i,j) = m(j,i)
            m(j,i) = t
 200     continue
 100  continue

      end
