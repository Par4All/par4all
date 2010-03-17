      subroutine stop_pause
      logical l1, l2, l3, l4
      x = rand()
      if(x.gt.0.) then 
         x=x+5.
         stop "finished"
      elseif(x.lt.-1.) then
         x=x+7.
         stop 3
      elseif(x.ne.-2.) then
         stop
      elseif(x.ge.-3.) then
         pause 10
      elseif(x.le.-4.) then
         pause
      elseif(x.eq.-5.) then
         continue
      elseif(.NOT. l1) then
         x = rand()
      elseif(l2.or.l3) then
 1       continue
      elseif(l3.and.l4) then
         x=x+1.
      endif
         
      end
