C     
C     
      PROGRAM EVAL
C
C     Programme de test pour l'evaluation partielle et le 
C     remplacement de constantes.
C
      INTEGER L(10)
C
C
      I=0
      J=1
      K=5
C
C     do not replace array references, but array indices
      L(I+J+1)=L(1)+L(J-0)+I
C
      DO I = J, K+0, J
         P=P+I
      ENDDO
C
      IF ((2*N*J-0) .GT. 0) THEN
         P=0
      END IF
C
C     test de PLUS et MINUS
      K= 2*M + 2*N
      K= 2*M - 2*N
      K= 2*M + (-2)*N
      K= -2*M + 2*N
      K= -2*M - 2*N
      K= -2*M + (-2)*N
C
      K= M + 2*N
      K= M - 2*N
      K= M +(-2)*N
      K= -M - 2*N
      K= -M +(-2)*N
      K= -M -(-2)*N
C
      K= 1 + N
      K= -1 -N
      K= -N -1
      K= -N + (-1)
      K= N + 1
      K= 0+N-0
C
C     test pour MULT
      K= 3*M*(-2)
      K= 1*N
      K= (2*N+1)*0
      K= (3*N)*M
      K= 2+((3*N+0)*(M-1) + 5)
      K= (N+1)*M
C
C     test pour DIV et MOD.
C     {A=3/2} est equivalent a {A=1}, mais different de {A=3./2}
      A= 3/2
      A= 3./2
      K= 3/2
      K= 4/2
      K= (-5)/(-2)
c     (-5)/2 = 5/(-2) = -2
      K= (-5)/2
      K= 5/(-2)
      K= (4*M+5)/2
      K= (4*M+5)/1
      K= (4*M+6)/2
c     MOD(3,2) = 1
      K1= MOD(3,2)
c     MOD(-3,-2) = -1
      K2= MOD(-3,-2)
C     Not true with gfortran
c     MOD(-3,2) = MOD(3,-2) = -1
c     MOD(-3,2) = -1
      K3= MOD(-3,2)
c     MOD(3,-2) = 1
      K4= MOD(3,-2)
      print *, k1, k2, k3, k4, "(should be 1, -1, -1, 1)"
C
C     do not replace written parameters of assign, fonctions 
C     or subroutines!
      I=10
      I=I
      J=0
      call fx(I, J)
C
      END

      subroutine fx(NR, NW)
      NW=NR+1
      end
