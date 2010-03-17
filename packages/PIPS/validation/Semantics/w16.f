      program w16

C     Derived from w15: remove I from the condition

C     Assignments in body:

      integer i

      i = 0

      do while(x.gt.0.)
         read *, y
         if(y.gt.0.) then
            i = 1
         endif
         read *, x
      enddo

      print *, i

      end
