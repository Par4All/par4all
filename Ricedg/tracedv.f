	program trace
	COMMON /X/ S
	real a, b
		a = 0
		do i=1,10
			b = 2 * a
			a = a + 1
		enddo
		S = 1
		do i=1,10
			call subA(S)
			call subB(S)
		enddo
		if (S.GE.1) then
			a = a * 2
		endif
    	END
	subroutine subA(N)
	Integer N
C	COMMON /X/ S
		N = 1
		call subC
		return
    	end
    	subroutine subB(M)
	Integer*2 M
    	real T
C    	COMMON /X/ S
		T = M+1
		call subD
		return
    	end
	subroutine subC
		return
	end
	subroutine subD
		return
	end

