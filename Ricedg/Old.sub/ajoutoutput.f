      program ajoutoutput
      integer a, c
      read *, c
      call ajoutoutputfoo(a, a, c)
      print *, a, c
      end
      subroutine ajoutoutputfoo(a, b, c)
      integer a, b, c
      a = 2 * c
      c = 2
      b = c * 5
      end
