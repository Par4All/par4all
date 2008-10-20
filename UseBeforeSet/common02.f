      program main
      
      call c1()
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
      
      
