C       Uninitialized scalar variables
	program scalar09
	integer x,y,z
	x = sqrt(2) -1
	if (x.gt.0 ) then
	   y = 1
	else
	   z = 1
	endif
	call foo(y)
	end
	subroutine foo(y)
	integer t
	t = sqrt(2) -1
	if (t.gt.1) then
	   y = 2
	else
	   t = 2
	endif
	print *,y
	end

