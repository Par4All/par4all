      program LOOP
      integer I,N,A(100),B(100)

      do i = 1, 100
         A(I) = 0.0
      enddo
      do i = 1, 100
         call PRIV1(A,B,i)
         call PRIV2(A,B,i)
      enddo
      end

      subroutine PRIV1(V,W,N)
      integer V(N),W,i
      integer WORK(100)
      save WORK
      
      do i = 1,N
         WORK(i) = V(i)
      enddo
      W = 0
      do i = 1,N
         W = W + WORK(n-i+1)
      enddo
      end

      subroutine PRIV2(V,W,N)
      integer V(N),W,i
      integer WORK(100),j
      common /toto/ WORK,j
      
      do i = 1,N
         WORK(i) = V(i)
      enddo
      W = 0
      do i = 1,N
         j = n-i+1
         W = W + WORK(j)
      enddo
      end
