      program main
      COMMON /W/ KMAX,KT
      integer I
      call A(I)
      end
      
      subroutine A(I)
      COMMON /W/ KMAX,KT
      integer I
C     commentaire pour statement
      KMAX = KMAX+2
      call B(I)
      call C3(I)
      call B(KMAX)
      return
      end       

      subroutine B(I)
      COMMON /W/ KMAX,KT
      integer I
      KT = KT +3
      call C1(I)
      call C2(I)
      return
      end 

      subroutine C1(I)
      COMMON /W/ KMAX,KT
      integer I
C     commentaire pour call
      call INC(KMAX)
      return
      end 

      subroutine C2(I)
      COMMON /W/ KMAX,KT
      integer I
      call INC(KT)
      return
      end 

      subroutine C3(I)
      COMMON /W/ KMAX, TT
      integer I
      call INC(TT)
      return
      end 

      subroutine INC(K)
      integer K
      K=K+1
      return
      end 


