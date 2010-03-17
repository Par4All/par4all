      subroutine unstructured01
      integer a,b
      if (a.eq.1) then
         if (b.eq.1) then 
            continue
         else
            PRINT *, "Error"
         endif
      endif
      end
