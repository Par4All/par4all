      program callgraph04

C     Make sure that recursive cycles are caught

      call foo

      end

      subroutine bar

      print *, "bar"

      call foo

      end

      subroutine foo

      print *, "foo"

      call bar

      end
