      program tiling03

C     Check tiling: use integer lower loop bounds and symbolic upper bounds

      real x(0:15, 0:10)

      do 100 i = 1, m
         do 200 j = 1, n
            x(i,j) = float(i+j)
 200     continue
 100  continue

      print *, x

      end
