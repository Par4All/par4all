      program PP
      integer A(100)

      call TOTO(A,100)
      call TITI(A,100)
      end

      subroutine TITI(V,N)
      integer V(N)
      common/com/B(2),C(3)

      do i =1,N
         V(i) = B(1)+C(3)
      enddo
      end

      subroutine TOTO(V,N)
      integer V(N),TITIV
      common/com/B(2),C(3)
      common/totocom/TOTOV(100)

      do i =1,N
         TOTOV(i) = i
         V(i) = B(1)+C(3)+TOTOV(100-i+1)
      enddo
      end
