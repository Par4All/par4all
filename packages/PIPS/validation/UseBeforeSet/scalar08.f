C       Uninitialized scalar variables
	program scalar08
	integer x,y,z,t
	logical c
	x = sqrt(2.0) -1
c	x = 1
	c = .false.
	t = sqrt(2.0)
	if (c.eq..true.) then 
           if ( x.gt.0 ) then
              y = 1
           else
	      z = 1
           endif
C          if c && x <=0 stop 
c	   print *,y
c	   y = 2
           y = y+1
	   y = 5
        endif
	if ( t.gt.1 ) then
           y = 3
        else
	   z = 1
        endif
c       if t <=1 && not( c&& x>0) stop 
c	print *,y
	do i=1,y
	   h = y
	enddo
	y = 6
	y = 7
	end

