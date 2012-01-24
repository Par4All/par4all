	subroutine transpose(M,N)
	real M(N,N)

	do i = 1, N-1
		do j = i+1, N
			M(i,j) = M(j,i)
		enddo
	enddo

	end
