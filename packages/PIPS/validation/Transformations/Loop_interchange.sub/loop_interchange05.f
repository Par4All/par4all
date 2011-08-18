!     Bug: make sure that label 200 and 300 are properly regenerated.

!     Non-Unit Stride Memory Access

      program loop_interchange05

      implicit none
      integer NUM
      parameter(NUM=5000)
      integer  i,j
      real     a(1:NUM,1:NUM), b(1:NUM,1:NUM), c(1:NUM,1:NUM), x

      a = 0
      b = 0
      c = 0

      read *, x

      do 300 i=1,NUM
         do 200 j=1,NUM
            if (x.gt.0.) go to 200
               c(j,i) = c(j,i) + a(j,i) * b(j,i)
 200     continue
 300  continue

      end
