program allocate03
  implicit none
  integer, dimension (:,:), allocatable :: a
  integer, dimension (:,:,:), allocatable :: b
  integer :: ierr
  integer :: size = 2
  allocate(a(size,size),stat=ierr)
  allocate(b(size,size,size),stat=ierr)
end program allocate03
