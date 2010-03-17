      program equiv24

C     Detect data bug: ANSI extension: overlapping initializations

      equivalence (m,n)

      data m /1/
      data n /1/

      print *, m

      end
