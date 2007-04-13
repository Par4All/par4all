      subroutine stack03(b, n)
      real a(n), b(n), c(10)
      data m /1/
      equivalence (a(1), m)

      print *, a, b, c

      end
