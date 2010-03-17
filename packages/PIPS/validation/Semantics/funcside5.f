      program funcside5
      integer foo5
      external foo5

C     Non compliant: definition of a variable in an expression using
C     this variable

      i1 = 10

      i1 = i1 + foo5(i1)

      print *, i1

      i1 = foo5(i1) + i1

      print *, i1

      end

      integer function foo5(j)
      j = j + 1
      foo5 = 2
      end
