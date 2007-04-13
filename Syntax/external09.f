      program external09

C      Same as external07, but for functions

C     Check the type consistency check option: PARSER_TYPE_CHECK_CALL_SITES
C     (it does not seem to work)
C     Check two parsing orders: F before or after EXTERNAL09
C     When F is called first, MakeAtom understands "call g(f)" as call "g(f())"

      external f
      integer f
      integer h

      call g(f)

      i = 2
      j = 3

      k = f(i, j)

      l = f(f(i,j), h())

      print *, k, l

      end

      integer function f(i, j)

      print *, "f called for ", i, j

      f = i+j

      end

      integer function h()
      h = 1
      end

      subroutine g(f)
      integer f
      external f
      print *, f(10,1)
      end

