C     Checking C prettyprint of Fortran code: how about labelless
C     continue?
      subroutine continue_test
      logical l1, l2, l3, l4

      read *,x
      if(x.gt.0.) then 
         continue
      elseif(x.lt.-1.) then
         continue
      else
c        x=x+1
      endif
      
      end
