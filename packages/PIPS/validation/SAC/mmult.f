      SUBROUTINE MMUL(a,b,c)
      INTEGER*4 SIZE
      PARAMETER (SIZE=4)


      INTEGER*4 a(SIZE,SIZE), b(SIZE,SIZE), c(SIZE,SIZE)
      INTEGER i, j,k

      DO 10  i=1, SIZE
            DO 20  j=1, SIZE
                  c(i,j)=0
                  DO 30  k=1, SIZE
                        c(i,j) = c(i,j) + a(i,k)*b(k,j)
30                ENDDO
20          ENDDO
10    ENDDO
      END
