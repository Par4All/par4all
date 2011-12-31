C       Uninitialized scalar variables
	program scalar14
	integer x,y,z
	x = sqrt(2) -1
	if (x.gt.0 ) then
	   z = y
	else
	   z = 1
	endif
	l = sqrt(5)
	if (l.gt.1) then
	   y = 2 
	else
	   print *,y
	endif
	print *,y
	end
