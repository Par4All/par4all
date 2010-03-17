      program callgraph06

C     Make sure that missing modules are not a problem: bar is missing here

      call foo

      end

      subroutine foo

      print *, "foo"

      call bar

      end
