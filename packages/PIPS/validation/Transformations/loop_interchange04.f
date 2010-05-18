!     Make sure that label 100 is not a problem for the loop interchange
!     nor for code regeneration

!     Non-Unit Stride Memory Access

      program loop_interchange04

      implicit none
      integer NUM
      parameter(NUM=5000)
      integer  i,j
      real     a(1:NUM,1:NUM), b(1:NUM,1:NUM), c(1:NUM,1:NUM)

      a = 0
      b = 0
      c = 0

      do 100 i=1,NUM
         do 100 j=1,NUM
            c(j,i) = c(j,i) + a(j,i) * b(j,i)
 100  continue

      end
