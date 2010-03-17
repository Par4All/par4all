! Equivalent stubs of Par4All runtime to keep the PIPS analyzes happy

	function P4A_COPY_TO_ACCEL(host_address, accel_address, n)
	integer n
	character host_address(n)
	character accel_address(n)
	integer i
	integer P4A_COPY_TO_ACCEL

	do i = 1,n
	   accel_address(i) = host_address(i)
	end do
	P4A_COPY_TO_ACCEL = accel_address
	end


	function P4A_COPY_FROM_ACCEL(host_address, accel_address, n)
	integer n
	character host_address(n)
	character accel_address(n)
	integer i
	integer P4A_COPY_FROM_ACCEL

	do i = 1,n
	   host_address(i) = accel_address(i)
	end do
	P4A_COPY_FROM_ACCEL = host_address
	end


	subroutine P4A_ACCEL_MALLOC(dest, n)
	integer n
	integer dest
! Do nothing since it is not representable in Fortran 77...
	end


	subroutine P4A_ACCEL_FREE(dest)
	integer dest
! Do nothing since it is not representable in Fortran 77...
	end
