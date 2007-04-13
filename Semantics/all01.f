C     Check that all static initialization are taken into account:
C     k==3 in OUTPUT

C     Check also that dead paths in the call graph are not used: 
C     l==1 in OUTPUT

      program all01
      common /bar/ k
      call foo1(i)
      print *, i, k
      end

      subroutine foo1(j)
      j = 1
      call output(j)
      end

      subroutine foo2(j)
      common /bar/ k
      data k /3/
      j = 2
      call output(j)
      end

      subroutine output(l)
      common /bar/ k
      print *, l, k
      end
