C       Uninitialized scalar variables
	program scalar11
	integer x,y
        x = sqrt(2.0) - 1
        if ( x.gt.0 ) then
           y = 1
        else
	   x = y
        endif
	end
