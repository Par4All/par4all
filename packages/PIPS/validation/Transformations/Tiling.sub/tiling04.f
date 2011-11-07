      program tiling04

C     Check tiling: use integer lower loop bounds and symbolic upper bounds

      real x(0:15, 0:10)

      do 100 i = m1, m2
         do 200 j = n1, n2
            x(i,j) = float(i+j)
 200     continue
 100  continue

      print *, x

      end
