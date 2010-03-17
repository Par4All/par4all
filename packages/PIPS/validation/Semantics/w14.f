      program w14

C     Derived from w12.

C     Conditional assignment in body:
      integer i

      i = 0

      read *, x
      do while(x.gt.0.)
         read *, y
         if(y.gt.0.) then
            i = 0
         endif
         read *, x
      enddo

      print *, i

      end
