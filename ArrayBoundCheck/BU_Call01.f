*
* Call with many array references
*
      SUBROUTINE RESID(U,V,R,N,A)
      INTEGER N
      REAL*8 U(N,N,N),V(N,N,N),R(N,N,N),A(0:3)
      INTEGER I3, I2, I1
C
      DO 600 I3=2,N-1
      DO 600 I2=2,N-1
      DO 600 I1=2,N-1
         R(I1,I2,I3)=V(I1,I2,I3)
     >      -A(0)*( U(I1,  I2,  I3  ) )
     >      -A(1)*( U(I1-1,I2,  I3  ) + U(I1+1,I2,  I3  )
     >                 +  U(I1,  I2-1,I3  ) + U(I1,  I2+1,I3  )
     >                 +  U(I1,  I2,  I3-1) + U(I1,  I2,  I3+1) )
     >      -A(2)*( U(I1-1,I2-1,I3  ) + U(I1+1,I2-1,I3  )
     >                 +  U(I1-1,I2+1,I3  ) + U(I1+1,I2+1,I3  )
     >                 +  U(I1,  I2-1,I3-1) + U(I1,  I2+1,I3-1)
     >                 +  U(I1,  I2-1,I3+1) + U(I1,  I2+1,I3+1)
     >                 +  U(I1-1,I2,  I3-1) + U(I1-1,I2,  I3+1)
     >                 +  U(I1+1,I2,  I3-1) + U(I1+1,I2,  I3+1) )
     >      -A(3)*( U(I1-1,I2-1,I3-1) + U(I1+1,I2-1,I3-1)
     >                 +  U(I1-1,I2+1,I3-1) + U(I1+1,I2+1,I3-1)
     >                 +  U(I1-1,I2-1,I3+1) + U(I1+1,I2-1,I3+1)
     >                 +  U(I1-1,I2+1,I3+1) + U(I1+1,I2+1,I3+1) )

 600  CONTINUE
C
C
      RETURN
      END
