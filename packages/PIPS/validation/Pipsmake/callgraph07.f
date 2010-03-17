      program callgraph07

C     Make sure that HEIGHT and DEPTH resources are recognized

      call foo

      end

      subroutine foo

      print *, "foo"

      call bar1
      call bar2

      end

      subroutine bar1

      print *, "bar"

      call bar2

      end

      subroutine bar2

      print *, "bar"

      end
