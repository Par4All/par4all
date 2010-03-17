      subroutine entry17(n)

C     Goal: check that an equivalence between static variables
C     is detected and results in a user error (for the time being).
C     I'm not convinced that the implementation of the necessary tricks
C     is worth the effort (a dummy variable per equivalence chain plus
C     equivalences)

C     Secondary goal: check that entry processing survives a previous call
C     to ParserError()

      data k /3/
      save m
      equivalence (m,k)

      print *, n

      return

      entry increment(n)

      m = k + 1

      end
