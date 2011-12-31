      program strdata01
      character*12 c
      data c /'hello world'/
      integer i
      data i /12/
      print *, c, i
      c = 'bye world'
      i = 3
      print *, c, i
      end
