      program unroll10

C     Bug Francois Ferrand: new loop variables should inherit the loop
C     index type exactly

      integer*4 I
      integer*2 J
      integer*4 K
      integer*8 L
      integer*2 A(0:199)
	      
      do 20 I = 0,199
         A(I) = I
 20   enddo
      print *, A
      end
