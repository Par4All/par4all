      program rice1      
      real a(1:101,1:101,1:101),x(1:101,1:101,1:101)

      do 100 i=1,100
         do 90 j=1,100
            do 30 k=1,100
               x(i,j+1,k) = a(i,j,k) +10
 30         continue
            do 80 l =1,50
               a(i+1,j,l)=x(i,j,l)+5
 80         continue
 90      continue
 100  continue
      end
