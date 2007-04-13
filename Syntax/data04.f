      subroutine data04

C     Detect data bug (or ANSI extension...)

      common m

      data m /1/

      print *, m

      end
