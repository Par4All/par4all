C       Uninitialized scalar variables
	program scalar06
	integer x,y
        x = sqrt(2.0) - 1
        if ( x.gt.0 ) then
           y = 1
        else
	   call sub(x,y)
        endif
	end
        subroutine sub(x,y)
	integer x,y
        x = y
        print *,x
        end
