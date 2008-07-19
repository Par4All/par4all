      program contreexemple
      integer a, c
      read *, c
      call contreexemplefoo(a, a, c)
      print *, a, c
      end
      subroutine contreexemplefoo(a, b ,c)
      integer a, b, c
      a = 2
      b = 3 * a
      c = a * b
      end






