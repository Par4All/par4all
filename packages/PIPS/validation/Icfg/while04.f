      program while04

C     Do we identify some WHILE control structures?

      do i = 1, n
         call foo(i)
         if(x.lt.eps) go to 200
         call bar(i)
      enddo

 200  continue

      end

      subroutine foo(i)
      end

      subroutine bar(i)
      end

