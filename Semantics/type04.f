      program type04

C     Goal: check the extension of the semantics analysis to logical
C     scalar variables

      logical l1, l2

      read *, l1

      l2 = l1 .AND. .NOT. l1

      read *, l1

      print *, l1, l2

      end

