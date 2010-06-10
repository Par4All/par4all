!     Copy of loopshared01.f, modified to be able to prettyprint the
!     loop body transformers of the first loop

      program loopshared02

      n = 0
      do 100 i = 1, 10
         if(.TRUE.) then
            j = i
            do 200 k = 1, 5
               l = k
               n = n + 1
 200        continue
         endif
 100  continue

!     n == 50
         print *, i, j, k, l, n

         end
