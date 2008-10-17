!
! Source code for module "foo" is not available but it can be generated
!
      program main
      print *, 'main'
      call bla(3)
      call foo(1, 'hello')
      end
      subroutine bla(i)
      print *, 'i=', i
      end
