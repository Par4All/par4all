      program data09

C     Detect data bug (or ANSI extension...)

      common m

      call init1
      call init2

      print *, m

      end

      subroutine init1

      common m

      data m /1/

      end

      subroutine init2

      common m

      data m /2/

      end
