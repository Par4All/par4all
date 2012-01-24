      subroutine induc2(a,n)
      real a(2*n)
      
      j = 1
      
      do 100 i = 1, n
         a(j) = a(j)+a(j-1)
         j = j + 2
 100  continue
      
      end
