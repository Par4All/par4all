      subroutine entry09(x, n)

C     Goal: make sure that implicitly static variables are properly processed

C     Secondary goal: make sure that the data statement appears only once because
C     l is declared in a common to mimic entry behavior

      complex x(n)
      character*1 y(m)

      data l /10/

      print *, x(1)

      return

      entry increment(y, m)

      print *, y(2)

      m = m + 1

      end
