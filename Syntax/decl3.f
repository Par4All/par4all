C     Ancien bug dans les declarations de
C     tableau de taille suppose'e (*).
C
C     Be'atrice Creusillet
C
C     Corrige le 19 de'cembre 1994
C
      program decl3
      real x(100)

      call toto(x(3))
      end

      subroutine toto(v)
      real v(3:*)
      
      v(3) = 0.
      end
