      subroutine data05

C     Detect data bug (or ANSI extension...)

      data m /1/

      common m

      print *, m

      end
