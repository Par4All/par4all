      program callgraph02

C     Make sure that height and depth are different

      call foo
      call bar

      end

      subroutine bar

      print *, "bar"

      call foo

      end

      subroutine foo

      print *, "foo"

      end
