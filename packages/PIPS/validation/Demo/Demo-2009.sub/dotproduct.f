      real function dotproduct(n,x,y)
      
      real x(n), y(n)
      real s
      integer i

      s=0.
      do  100 i =1,n
         s= s+x(i)*y(i)
 100  continue

      dotproduct =s
      end

      program demo1

      parameter (idim = 90)
      real  v1(idim), v2(idim), result

      do i = 1,idim
         v1(i) = float(i)
         v2(i) = 2.
      enddo

      result = dotproduct(idim,v1,v2)
      print *,"result =", result
      end
      
      
