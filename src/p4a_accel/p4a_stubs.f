!> @addtogroup p4a_accel_stubs

!> @{

!> @defgroup p4a_accel_Fortran_stubs Equivalent stubs in Fortran of Par4All runtime to have PIPS analyzes happy

!> @{

!>  Stub for copying memory from the host to the hardware accelerator.
!>
!>  Since it is a stub so that PIPS can understand it, use simple
!>  implementation with standard memory copy operations
!>
!>  Do not change the place of the pointers in the API. The host address
!>  is always in the first position...
!>
!>  @param[in] host_address is the address of a source zone in the host memory
!>
!>  @param[out] accel_address is the address of a destination zone in the
!>  accelerator memory
!>
!>  @param[in] size is the size in bytes of the memory zone to copy
	function P4A_COPY_TO_ACCEL(host_address, accel_address, size)
	integer size
	character host_address(size)
	character accel_address(size)
	integer i
	character P4A_COPY_TO_ACCEL(size)

	do i = 1,size
	   accel_address(i) = host_address(i)
	end do
	P4A_COPY_TO_ACCEL = accel_address
	end

!>  Stub for copying memory from the hardware accelerator to the host.
!>
!>  Do not change the place of the pointers in the API. The host address
!>  is always in the first position...
!>
!>  @param[out] host_address is the address of a destination zone in the
!>  host memory
!>
!>  @param[in] accel_address is the address of a source zone in the
!>  accelerator memory
!>
!>  @param[in] size is the size in bytes of the memory zone to copy
	function P4A_COPY_FROM_ACCEL(host_address, accel_address, size)
	integer size
	character host_address(size)
	character accel_address(size)
	integer i
	character P4A_COPY_FROM_ACCEL(size)

	do i = 1,size
	   host_address(i) = accel_address(i)
	end do
	P4A_COPY_FROM_ACCEL = host_address
	end


!>  Stub for allocating memory on the hardware accelerator.
!>
!>  @param[out] address is the address of a variable that is updated by
!>  this macro to contains the address of the allocated memory block
!>  @param[in] size is the size to allocate in bytes
	subroutine P4A_ACCEL_MALLOC(address, size)
	integer size
	integer address
! Do nothing since it is not representable in Fortran 77...
	end


!>  Stub for freeing memory on the hardware accelerator.
!>
!>  @param[in] address is the address of a previously allocated memory
!>  zone on the hardware accelerator
	subroutine P4A_ACCEL_FREE(address)
	integer address
! Do nothing since it is not representable in Fortran 77...
	end
!> @}
!> @}
