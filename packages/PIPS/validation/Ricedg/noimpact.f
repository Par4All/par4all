      program noimpact
      integer a, c, d
      read *, a, c, d
      call noimpactfoo(a, a, c, d)
      print *, a, c, d
      end
      subroutine noimpactfoo(a, b, c, d)
      integer a, b, c, d
      c = a * 2
      d = c + 2
      b = d * 3
      end
