      program data02

C     Variable I is initialized in foo but is not visible
C     in data02. Its value should be eliminated from the
C     initial precondition.

      call foo

      j = 2
      print *, j

      end

      subroutine foo

      data i /1/

      k = 3

      end
