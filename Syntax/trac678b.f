!     Floating point sizing	
      
      program main
      
! This is not OK with gfortran and no implicit integer conversion is performed
      integer array(2.456)
      
      array(1) = 2
      print *, array(1)
      
      end
