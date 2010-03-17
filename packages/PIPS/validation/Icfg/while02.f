      program while02

C     Do we identify some WHILE control structures?

      i = 0
 100  continue
      call foo
      if(x.lt.eps) go to 200
      i = i + 1
      go to 100

 200   continue

      end

      subroutine foo
      end

