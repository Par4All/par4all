      program main
      c = .true.
      if (c) then
         call p
      else
         call q
      endif
      end
      subroutine p
      common w
      t =.true.
      if (t) then
         w = 4
      endif
      end
      subroutine q
      common v
      print *,v
      end
      
      
