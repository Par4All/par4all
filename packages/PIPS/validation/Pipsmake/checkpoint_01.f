      program checkpoint
      integer i
      i = 1
      print *, 'checkpoint', i
      call foo
      end

      subroutine foo
      integer j
      j = 2
      print *, 'foo', j
      call bla
      end

      subroutine bla
      integer k
      k = 3
      print *, 'bla', k
      end
