C     non unit loop increment.
C
      program INCR3
      integer I, A(12)
      do I = 1,10,3
         A(I) = 1
         A(I+1) = 2
         A(I+2) = 3
      enddo
      print *, A
      end
