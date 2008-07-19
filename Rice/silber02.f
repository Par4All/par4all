      subroutine silber02(a, b, n)

C     Check what happens with coarse grain parallelization on Georges' examples
C     presented at CPC'98

      real a(n), b(n)

      do i = 1, n
         if(a(i-1).ge.0.) then
            b(i) = 0.
         endif
         a(i) = 1.
      enddo

      end
