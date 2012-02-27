      program callgraph05

C     Make sure that recursive cycles are caught but not regular calls

C     Well, it is not working: all calls are considered part of the cycle

      call foo

      end

      subroutine bar

      print *, "bar"

      call foo

      call bar2

      end

      subroutine foo

      print *, "foo"

      call bar

      call foo2

      end

      subroutine bar2

      print *, "bar2"

      end

      subroutine foo2

      print *, "foo2"

      end

