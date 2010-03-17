      subroutine altret04(x, *)

C     Check that the block returned for the call with implicit return
C     is properly handled in spite of the logical IF

      if(x.gt.0.) call bar2(*123, x, *234)

      return

 123  continue
      return 1

c     Second return
 234  return 2

      end

      subroutine bar2(*, x, *)

      x = x + 1.

      if(x.lt.0) then  
         return 1
      elseif(x.gt.0.) then 
         return 2
      else 
         return
      endif

      end
