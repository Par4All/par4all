C       Uninitialized scalar variables
	program scalar07
	common /foo/ a,b
	integer x,y
        x = sqrt(2.0) - 1
	print *,m
        if ( x.gt.0 ) then
           y = 1
	   call sub(x,y)
        else
	   call sub(x,y)
        endif
	end
        subroutine sub(x,y)
	common /foo/ a,b
	integer x,y
	a = b 
        x = y
	l = h
        print *,x
        end

