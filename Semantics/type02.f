      program type02

C     Goal: check the extension of the semantics analysis to logical
C     scalar variables

      logical l1, l2, l3, l4

      l1 = .TRUE.
      l2 = .FALSE.
      l1 = .NOT. l2
      l1 = .NOT. l1
      l3 = l1 .OR. l2
      l4 = l1. AND. l2

      print *, l1, l2, l3, l4

      end

