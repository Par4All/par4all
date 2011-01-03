	program bar
        do i = 1, 10
           if i .eq. 2 goto 1
	   goto 1
	end do
 1	print 'Coucou'
	end
