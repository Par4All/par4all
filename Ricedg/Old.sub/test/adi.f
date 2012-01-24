      program adi
      integer i,j,iter 
      parameter (N = 100)
      parameter (MAXI = 1000)
      integer N,MAXI 
      real c(N,N), a(N,N), b(N,N)
      
      DO iter = 1, MAXI
!NESTOR$ SINGLE
         DO j = 2, N
            DO i = 1, N
               c(i,j) = c(i,j)-c(i,j-1)*a(i,j)/b(i,j-1)
               b(i,j) = b(i,j)-a(i,j)*a(i,j)/b(i,j-1)
            ENDDO
         ENDDO

!NESTOR$ SINGLE
         DO i = 1,N
            c(i,N) = c(i,N)/b(i,N)
         ENDDO

!NESTOR$ SINGLE
         DO j = N-1,1,-1
            DO i = 2,N
               c(i,j) = (c(i,j)-a(i,j+1)*c(i,j+1))/b(i,j)
            ENDDO
         ENDDO

!NESTOR$ SINGLE
         DO j = 2, N
            DO i = 1, N
               c(i,j) = c(i,j)-c(i-1,j)*a(i,j)/b(i-1,j)
               b(i,j) = b(i,j)-a(i,j)*a(i,j)/b(i-1,j)
            ENDDO
         ENDDO

!NESTOR$ SINGLE
         DO j = 1,N
            c(N,j) = c(N,j)/b(N,j)
         ENDDO

!NESTOR$ SINGLE
         DO j = N-1,1,-1
            DO i = 2,N
               c(i,j) = (c(i,j)-a(i+1,j)*c(i+1,j))/b(i,j)
            ENDDO
         ENDDO
         
      ENDDO

      end
! write c,a,b
