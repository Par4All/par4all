C       Uninitialized scalar variables
	program scalar03
	integer x,y
	call sub(x,y)
        print * ,x,y
	end
        subroutine sub(x,y)
	integer x,y
        x=y
        end
