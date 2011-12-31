      program strdata02
      character*12 c
      data c /'hello world'/
      integer i
      data i /12/
      real*4 r
      data r /1.0/
      logical l
      data l /.true./
      print *, c, i, r, l
      c = 'bye world'
      i = 3
      r = 2.0
      l = .false.
      print *, c, i, r, l
      end
