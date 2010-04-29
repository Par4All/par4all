C     Excerpt of eval.f: bug for partial evaluation of subscript expression
C     
      PROGRAM EVAL05
C
C     Programme de test pour l'evaluation partielle et le 
C     remplacement de constantes.
C
      INTEGER L(10)
C
C
      I=0
      J=1
C
C     do not replace array references, but array indices
      L(I+J+1)=L(1)+L(J-0)+I
      END
