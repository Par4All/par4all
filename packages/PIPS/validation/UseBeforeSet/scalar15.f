C       Uninitialized scalar variables
	program scalar15
	integer x(5),y
	call sub(2*x(1)+3*y,y)
	end
        subroutine sub(f1,f2)
	integer f1,f2
	k = f1
c	f1 = k
        end
