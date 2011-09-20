C step_sizetype
C
C Problem Fortran allows to use different size of integers
C with compilation option
C Thus the size of the Fortran INTEGER cannot be determined at
C compilation time
C
C We consider Fortran INTEGER as STEP_INTEGER and then
C determine the size of STEP_INTEGER called kind_type
C using the primitive kind
C
C This is only necessary for Fortran.
C
      SUBROUTINE step_sizetype(step_type, kind_type)
      implicit none
      include "STEP.h"
      INTEGER*4 step_type, kind_type
      integer i
      real r
      double precision d
      complex c
      logical l

      select case (step_type)
         case (STEP_INTEGER)
            kind_type = kind(i)
         case (STEP_REAL)
            kind_type = kind(r)
         case (STEP_DOUBLE_PRECISION)
            kind_type = kind(d)
         case (STEP_COMPLEX)
            kind_type = 2*kind(c)
         case default
            kind_type = 0
      end select
      END
