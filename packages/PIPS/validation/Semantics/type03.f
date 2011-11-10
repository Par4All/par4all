      program type03

C     Goal: check the extension of the semantics analysis to logical
C     scalar variables

      logical l1, l2, l3, l4, l5, l6

      l3 = l1 .OR. l2
      l4 = l1 .AND. l2
      l5 = l1 .EQV. l2
      l6 = l1 .NEQV. l2

      print *, l1, l2, l3, l4, l5, l6

      end

