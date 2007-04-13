      program w13

C     Derived from w12. ELSEIF added to obtain unfeasible paths

C     Assignments in body: check that the initial condition I=0 and the
C     body exit condition 0<=I<=1 are merged as entry condition for the
C     loop body before the fixpoint is computed.

C     To obtain a precise loop transformer, the transformer for the
C     entered loop and the transformer for the non-entered loop should
C     be computed and their convex hull taken, but it is not useful,
C     since the operation is performed when preconditions are computed.

      integer i

      i = 0

      do while(x.gt.0.)
         if(i.eq.1) then
            i = 0
         elseif(i.eq.0)then
            i = 1
         endif
         read *, x
      enddo

      print *, i

      end
