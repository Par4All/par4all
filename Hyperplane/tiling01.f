      program tiling01

C     Check tiling: use integer loop bounds, mutiple of the tile size

      real x(0:15, 0:10)

      do 100 i = 1, 15
         do 200 j = 1, 10
            x(i,j) = float(i+j)
 200     continue
 100  continue

      print *, x

      end
