	PROGRAM if

	parameter (N=100)
	integer a(N,N,N)
	integer b(N,N,N)
	integer c(N,N,N)
	
        real delta, actual
   
C        actual = secnds(0.0)

	do i = 1,n
	  do j = 1,n
            do k = 1,n
               a(i,j,k) = n
            enddo
          enddo
        enddo

!NESTOR$ SINGLE
	do i = 1,n
	  do j = 2,n
	    do k = 1, a(i,j+1,1)
	    	a(i,j,k) = b(i,j,k) + a(i,j-1,k)
	    	c(i,j,k) = a(i,j,k)**2
	    enddo
	  enddo  
	enddo

C        delta = secnds(actual)

C        write(6,*) 'Le temps de calcul est:=',delta,'s'
	end
