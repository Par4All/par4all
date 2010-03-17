      program suppress02

C     Check suppress_dead_code for string variables

      if('a'.gt.'b') then
         print *, 'a is greater than b'
      else
         print *, 'b is greater than a'
      endif

      end
