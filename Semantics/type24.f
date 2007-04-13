      program type24

C     No substring operator is applied interprocedurally although s1 and
C     s2 should receive different values

      character*20 s1
      character*2 s2

      call foo(s1)
      call foo(s2)

      print *, s1, s2

      end

      subroutine foo(s)

C     Variable and constant lengths must be taken into account and
C     substring applied... except when you cannot and have to assume
C     that the assignment is legal

      character*(*) s

C     Should not be truncated since the actual lenght of s is unknown
      s = "Hello World!"

      end
