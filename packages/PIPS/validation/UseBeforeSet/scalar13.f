C       Uninitialized scalar variables
	program scalar13
	integer x,y,z
	x = sqrt(2) -1
	if (x.gt.0 ) then
	   z = y
	else
	   z = 1
	endif
	l = sqrt(5)
	if (l.gt.1) then
	   x = 2 
	else
	   print *,y
	endif
	end
