      program supflow
      integer a, c, d
      read *, a
      call supflowfoo(a, a, c, d)
      print *, a, c, d
      end
      subroutine supflowfoo(a, b, c, d)
      integer a, b, c, d
      a = 2
      c = b * a
      d = a + c
      end













