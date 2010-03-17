C     Bug dans les declarations signales par Benoit de Dinechin
C
C     Francois Irigoin
C
C     Corrige (?) le 5 avril 1991
C
      program decl
      common /toto/ x(2)
      integer x

      x(1) = 0
      end
