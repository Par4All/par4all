      program w15

C     Derived from w12 and w13. ELSE removed to obtain unfeasible paths

C     Assignments in body:

      integer i

      i = 0

      do while(x.gt.0.)
         if(i.eq.0) then
            i = 1
         endif
         read *, x
      enddo

      print *, i

      end
