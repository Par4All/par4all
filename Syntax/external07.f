      program external07

C     Check the type consistency check option: PARSER_TYPE_CHECK_CALL_SITES
C     (it does not seem to work)
C     Check two parsing orders: F before or after EXTERNAL07
C     When F is called first, MakeAtom understands "call g(f)" as call "g(f())"

      external f

      call g(f)

      call f(i, j)

      call h

      end

      subroutine f(i)

      print *, i

      end

      subroutine h
      stop
      end
