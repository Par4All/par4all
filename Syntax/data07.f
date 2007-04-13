      subroutine data07

C     Detect data bug (or ANSI extension...)

      common /nga/ m

      data m /1/

      print *, m

      end
