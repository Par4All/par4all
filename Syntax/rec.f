      subroutine rec(n, m)

      if(n.eq.0) then
         m = 1
      else
         call rec(n-1, m)
         m = n*m
      endif

      end


      
