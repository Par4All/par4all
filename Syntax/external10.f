C     When f is parsed, you cannot decide yet

      subroutine external10(f)

      external f

      call g(f)

      end

      subroutine f

      print *, "Hello world!"

      end
