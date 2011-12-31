C       Uninitialized scalar variables
	program scalar12
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
	call foofoo(y)
	print *,y
	end
	subroutine foofoo(y)
	integer l
	read *,l
	if (l.gt.1) then
	   y = 2 
	else
	   print *,y
	endif
	end
