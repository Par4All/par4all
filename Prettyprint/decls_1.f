! clean declarations
      program decls1
      integer a, b, c, t(3)
      common // a, b, c
      equivalence (t(1),a)
      print *, a
      call foo
      call bla
      end

      subroutine foo
      integer a, b, c, t(3)
      common // a, b, c
      equivalence (t(1),a)
      data a, b, c / 1, 2, 3 /
      print *, b
      end

      subroutine bla
      integer a, b, c, t(3)
      common // a, b, c
      equivalence (t(1),a)
      print *, t(3)
      end
