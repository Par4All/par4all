      subroutine entry18(n)

C     Goal: check that multiple entries each have a proper entry common
C     for multiple static variables (and a unique DATA, see also entry16.f)

      data k /3/
      integer m(10)
      save m, l

      print *, m(1)+n, k+l

      return

      entry increment1(n)

      m(1) = n + 1

      entry increment2(n)

      m(2) = n + 1

      entry increment3(n)

      m(3) = n + 1

      entry increment4(n)

      m(4) = n + 1

      end
