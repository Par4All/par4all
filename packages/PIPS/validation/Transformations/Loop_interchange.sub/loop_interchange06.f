!     Bug: make sure that labels 100, 200 and 300 are properly regenerated.

      program loop_interchange06

      implicit none
      integer NUM
      parameter(NUM=50)
      integer  i,j, k
      real     a(1:NUM,1:NUM,1:NUM), b(1:NUM,1:NUM,1:NUM),
     &     c(1:NUM,1:NUM,1:NUM)
      real x

      a = 0
      b = 0
      c = 0

      do 300 i = 1, NUM
         do 200 j = 1, NUM
            do 100 k = 1, NUM
               if(x.gt.0.) go to 100
               c(k,j,i) = c(k,j,i) + a(k,j,i) * b(k,j,i)
 100        continue
 200     continue
 300  continue

      end
