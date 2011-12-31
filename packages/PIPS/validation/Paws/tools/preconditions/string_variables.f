      program strdata01

!     Use advanced mode!
!     It requires SEMANCTICS_ANALYZE_SCALAR_STRING_VARIABLES property to be set.

      character*12 c
      data c /'hello world'/
      integer i
      data i /12/
      print *, c, i
      c = 'bye world'
      i = 3
      print *, c, i
      end
