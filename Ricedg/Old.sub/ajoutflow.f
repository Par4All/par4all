      program ajoutflow
      integer a, c
      read *, a, c
      call ajoutflowfoo(a, a, c)
      print *, a, c
      end
      subroutine ajoutflowfoo(a, b, c)
      integer a, b, c
      a = 2 * c
      c = 2 + b
      end




























