      program loopinit4

C     Like loopinit2, but computing transformer in context

c     Check that loop index initialization is performed correctly: the initial
c     value of i is preserved by the loop

c Not handled properly by PIPS: inc(j) is evaluated when the initial value 
c of index i is computed (this is an assignment and the user call is detected), 
c but it is not evaluated when the loop range is c computed (because only affine
c ranges are taken into account)!

C     A fix: add an intermediate value and use it to call inc(j), use it as a
c     lower loop bound.

      real t(10)

      j = 2

      n = 0

      do i = inc(j), n, 1
         t(i) = 0.
         j = j + 2
      enddo

c     The precondition should be: i==3, j==3
      print *, i, j

      end
      
      integer function inc(k)
      k = k + 1
      inc = k
      end
