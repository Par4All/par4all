      program common1
C     test du parsing de deux apparitions du meme common
C     il faudrait verifier qu'ils ont bien la meme taille
      common /toto/ x, y
      call foo
      end
      subroutine foo
      common /toto/ x, y
      end
