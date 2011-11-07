      SUBROUTINE TILING(A,B)
      
      INTEGER T,SZ
      PARAMETER (T=10,SZ=100)
      INTEGER A(1:SZ,1:SZ)
      INTEGER B(0:SZ+1,0:SZ+1)
      INTEGER kernel(0:2,0:2)

      DO i=1,SZ,T
        DO j=1,SZ,T
          DO k=i,i+T-1
            DO l=j,j+T-1
              A(k,l)=0
              DO m=-1,1
                DO n=-1,1
                  A(k,l) = A(k,l) + B(k+m,l+n)*kernel(m+1,n+1)
                ENDDO
              ENDDO
            ENDDO
          ENDDO
        ENDDO
      ENDDO

      DO i=1,SZ
        DO j=1,SZ
          print *,A(i,j)
        ENDDO
      ENDDO

      END
