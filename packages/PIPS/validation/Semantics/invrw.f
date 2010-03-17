      program invrw

C     Check the computation of invariant r+w==0

      integer r, w

      r = 0
      w = 0

      do i = 1, l
         if(x.gt.0) then
            r = r + 1
            w = w - 1
         else
            r = r - 1
            w = w + 1
         endif
      enddo

      print *, i, l, r, w

      end
