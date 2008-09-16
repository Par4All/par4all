      program unstr5
      integer i
      
	  i = 4

      if(i.ge.3) then
         i = 3
         goto 100
      else
         i = 1
         goto 100
      endif
100   print *, j
      
      end
