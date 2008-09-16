! testing callgraph generation
      program p
      call c1
      call c2
      call c2
      call c3
      end

      subroutine c1
      call c3
      end

      subroutine c2
      call c4
      end
      
      subroutine c3
      end

      subroutine c4
      call c1
      end
