C       Uninitialized scalar variables
	program scalar05
	integer x,y
        common /aaa/ x,y
	x = y
	print *,x
	end
