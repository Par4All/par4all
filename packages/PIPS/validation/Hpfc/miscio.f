! miscellaneous I/Os
      program miscio

      integer n
      parameter (n=10)

      real a(n), b(n)

!hpf$ processors p(2)
!hpf$ distribute a(block) onto p

      integer i

      print *, 'read b(n)'
! update a shared array
      read *, (b(i), i=1, n)

      print *, 'print b(n)'
! collect a shared array
      print *, (b(i), i=1, n)

      print *, 'read a(n)'
! update a distributed array
      read *, (a(i), i=1, n)

      print *, 'print a(n)'
! collect a distributed array
      print *, (a(i), i=1, n)

      print *, 'done'
      end
