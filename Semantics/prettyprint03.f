! Check constraint sort (bug spotted in Semantics/boolean07)

      program prettyprint03

      if(j.le.i+1.and.i.le.j) then
         j = 1
      endif

      if(i.le.j.and.j.le.i+1) then
         j = 2
      endif

      end
