C     Check that the formal parameter lists are correct (bug found in
C     Perfect/TRACK by CA)

      subroutine entry20(n)

c     A variable cannot be used before (textually) it is declared as a
c     formal parameter unless it is in a type statement, but PIPS cannot
c     tell if the occurence was in a type statement or in an executable
c     statement: a simple warning is printed for k (legal) and m (illegal)

      real k

      m = 0

      print *, n

      return

      entry increment(m, k)

      m = k + 1

      end
