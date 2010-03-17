      PROGRAM parentheses

C     Check unary operator - and PRETTYPRINT_ALL_PARENTHESES property

      INTEGER A,B,C,D,E
      
c     expression avec parentheses pour forcer les multiply-add
      A = (60 * B + (-16 * (C+D) + E ))

c     sortie du prettyprinter : A = 60*B+-16*(C+D)+E 

      end
