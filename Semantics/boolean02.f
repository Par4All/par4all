	program boolean02
	logical b1, b2, b3, b4, b5, b6, b7
	b1 = .true.
	b2 = .false.
	read *, b3
	b4 = b1.AND.b3
	b5 = b2.OR.b3
	b6 = .NOT.b3
! XOR is missing
! http://gcc.gnu.org/onlinedocs/gfortran/XOR.html
!	b7 = XOR(b3,b2)
	print *, b1, b2, b3, b4, b5, b6, b7
	end
