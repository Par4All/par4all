      subroutine all02(i)
      integer i
      common /foo/ j
      data j /1/
      i = 2
      print *, i, j

      call bla2(i)
      end

      subroutine bla2(i)
      integer i
      print *, i
      end
