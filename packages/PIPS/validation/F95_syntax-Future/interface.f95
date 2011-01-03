program test
  use iso_c_binding
  interface
     subroutine foo(val) bind(C)
       use iso_c_binding
       integer, VALUE :: val
     end subroutine foo
  end interface
  integer :: i
  i = 5
  call foo (i)
end program test
