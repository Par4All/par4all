      program data03

C     Check triangular data initialization: PIPS parser used to core
C     dump It does not anymore but the initial value of k is lost with
C     the current implementation (June 2002).

      real x(100,100)

      data m, l, ((x(i,j), i = 1, j), j = 1, 10), k /-1, 56*1., 3/

      print*, x(2,4), k, l, m

      end
