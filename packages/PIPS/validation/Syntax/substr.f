      program substr
      character*10 f
      data f /'abcdefghij'/

      print *, f
      print *, f(1:10)
      print *, f(:9)
      print *, f(2:)
      print *, f(2:9)

      end
