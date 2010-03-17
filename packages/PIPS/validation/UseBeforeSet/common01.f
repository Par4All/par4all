      program main     
      call p
      end
      subroutine p
      common w
      logical t
      t =.true.
      if (t) then
         call q
      endif
      end
      subroutine q
      common v
      print *,v
      end
      
      
