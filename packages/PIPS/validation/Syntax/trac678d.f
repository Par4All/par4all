!     Floating point initialization of integer parameter

!     Make sure partial eval is OK
      
      program main
      
      integer*4 i
      parameter ( i = 100 )
      integer*4 j
      parameter ( j = i*0.1 )

      integer*4 array

      common / C1 / 
     *     array(j)

      array(1) = 2*j
      print *,array(1)

      end
