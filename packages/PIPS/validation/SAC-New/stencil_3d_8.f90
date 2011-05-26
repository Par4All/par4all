! sum all the elements of the array
real function sum_array (array, n1, n2, n3)
  implicit none
  integer :: n1,n2,n3,i1,i2,i3
  real array(n1,n2,n3)

  do i3=1, n3
     do i2=1, n2
        do i1=1, n1
           sum_array = sum_array + array(i1,i2,i3)
        end do
     end do
  end do
end function sum_array

program main
  implicit none
  integer :: n1,n2,n3,L,is1,ie1,is2,ie2,is3,ie3
  parameter(L=4, n1=100, n2=100, n3=100)
  real, dimension(:,:,:), allocatable :: u,v
  real, dimension(-L:L) :: c
  real result, sum_array

  is1=1;ie1=n1
  is2=1;ie2=n2
  is3=1;ie3=n3

  c = 3.

  allocate(v(n1,n2,n3))
  allocate(u(n1,n2,n3))

  ! Simple case
  u = 1.; v = 1.;
  call stencil8(u,v,c,n1,n2,n3,is1,ie1,is2,ie2,is3,ie3)
  result = sum_array (u, n1, n2, n3)
  PRINT *, ("the sum is"), result
  deallocate(v)
  deallocate(u)

end program main

subroutine stencil8(u,v,c,n1,n2,n3,is1,ie1,is2,ie2,is3,ie3)
  ! Stencil length : 2*L

  implicit none
  integer :: i1,i2,i3,is1,ie1,is2,ie2,is3,ie3,n1,n2,n3,L
  parameter(L=4)
  real, dimension(n1,n2,n3) :: u, v
  real, dimension(-L:L) :: c
  real c_4,c_3,c_1, c_2, c0, c1, c2,c3,c4

  c_4 = c(-4); c_3 = c(-3); c_2 = c(-2); c_1 = c(-1);
  c0 = c(0);
  c4 = c(4); c3 = c(3); c2 = c(2); c1 = c(1);

  do i3=is3+L,ie3-L
     do i2=is2+L,ie2-L
        do i1=is1+L,ie1-L
           u(i1,i2,i3) = &
                + c_4 * (v(i1-4,i2,i3) + v(i1,i2-4,i3) + v(i1,i2,i3-4))&
                + c_3 * (v(i1-3,i2,i3) + v(i1,i2-3,i3) + v(i1,i2,i3-3))&
                + c_2 * (v(i1-2,i2,i3) + v(i1,i2-2,i3) + v(i1,i2,i3-2))&
                + c_1 * (v(i1-1,i2,i3) + v(i1,i2-1,i3) + v(i1,i2,i3-1))&
                + c0  *  v(i1,  i2,i3) * 3&
                + c1  * (v(i1+1,i2,i3) + v(i1,i2+1,i3) + v(i1,i2,i3+1))&
                + c2  * (v(i1+2,i2,i3) + v(i1,i2+2,i3) + v(i1,i2,i3+2))&
                + c3  * (v(i1+3,i2,i3) + v(i1,i2+3,i3) + v(i1,i2,i3+3))&
                + c4  * (v(i1+4,i2,i3) + v(i1,i2+4,i3) + v(i1,i2,i3+4))
        end do
     end do
  end do

end subroutine stencil8
