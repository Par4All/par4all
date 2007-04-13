      subroutine comments3(t,n)
      real t(n)

c     Do we lose comments in front of an ENDDO?

c     No, as long as you do not parallelize the loop...

      do i = 1, N
         t(i) = 0.
c     this comment might be lost!
      enddo

      end
