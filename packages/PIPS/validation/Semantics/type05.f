      program type05

C     Goal: check the extension of the semantics analysis to logical
C     scalar variables in tests

      logical l1, l2

      read *, l1

      if(l1) then
         l2 = .FALSE.
         print *, l1, l2
      endif

      print *, l1, l2

      end
