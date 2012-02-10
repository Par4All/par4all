! Check constraint sort (bug spotted in Semantics/altret01)

      program prettyprint02

      if(1.le.n.and.n.le.2) then
         j = 1
      endif

      if(n.le.2.and.1.le.n) then
         j = 2
      endif

      end

