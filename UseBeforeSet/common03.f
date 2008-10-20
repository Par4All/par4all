      program main
      t = .true.
      if (t) then 
         call c1()
      endif
      call c2()
      end
      subroutine c1
      common w
      w = 5
      end
      subroutine c2
      common v
      print *,v
      end
      
      
